import math
import os

import numpy as np

from appearance.textures import DNA_BASE_COLORS, RNA_BASE_COLORS, get_texture
from geometry_nodes.geometry_nodes_modifier import GeometryNodesModifier
from geometry_nodes.nodes import Points, InputValue, InstanceOnPoints, JoinGeometry, \
    create_geometry_line, RealizeInstances, Position, make_function, Index, SetMaterial, \
    RepeatZone, StoredNamedAttribute, NamedAttribute, VectorMath, TransformGeometry, InputVector, MeshLine, BooleanMath, \
    Simulation, MathNode, CombineXYZ, Switch, CylinderMesh, ConeMesh, Frame, SeparateXYZ, ForEachZone, Quadrilateral, \
    InputInteger, \
    CurveWireFrame, ValueToString, StringToCurves, FillCurve, CompareNode, \
    InputMaterial, BoundingBox, InputString, SampleIndex, StringJoin, SliceString, Reroute, CharToAscii, \
    StringLength, IntegerMath, FindInString, SetPosition, ImportCSV, NodeGroup, \
    DomainSize, CombineBundle, SeparateBundle, GetBundleItem, SceneTime, ExtrudeMesh, \
    IndexSwitch, CurveLine, ResampleCurve, InputTangent, CaptureAttribute, DuplicateElements, \
    RandomValue, AlignRotationToVector, IcoSphere, MeshToCurve, CurveCircle, CurveToMesh, UVSphere, SetShadeSmooth, \
    SampleCurve, PointsToCurve, SetCurveNormal, VectorRotate, MapRange, MixNode, AccumulateField, \
    CurveLength, SeparateGeometry, InputRotation, MorphNode, SetSplineCyclic, \
    Grid, GeometryToInstance, RotateInstances, DeleteGeometry, \
    MorphNode2, SetCurveRadius, SplineParameter, AttributeStatistic, \
    TranslateInstances, ScaleInstances, MeshToCurve
from interface.ibpy import Vector
from objects.logo import logo_curve
from utils.constants import DATA_DIR, FRAME_RATE
from utils.kwargs import get_from_kwargs

pi = math.pi


def csv_column(path, default="Value"):
    """The name of the single column of a one-column csv file.

    Blender's ``Import CSV`` node always reads the first line of the file as
    the header and names the point attribute after it, so a file of bare
    numbers loses its first value and gets an attribute called ``110``. Reading
    the header here keeps the geometry nodes in step with whatever the file
    actually says, and a header that is itself a number is worth a warning:
    that file has no header and its first row will be missing from the tape.

    :param path: the csv file, read at build time.
    :param default: what to call the column if the file cannot be read.
    :return: the column name to hand to a ``Named Attribute`` node.
    """
    try:
        with open(path) as file:
            header = file.readline().strip().split(",")[0].strip()
    except OSError:
        print("Cannot read the header of " + str(path) + ", assuming '" + default + "'")
        return default
    if not header:
        return default
    try:
        float(header)
    except ValueError:
        return header
    print("Warning: " + str(path) + " starts with the number '" + header
          + "'. Import CSV takes the first line as the header, so this value "
            "will not appear on the tape. Add a header line to the file.")
    return header


# ---------------------------------------------------------------------------
# The instruction palette
# ---------------------------------------------------------------------------
# One colour per brainfuck instruction, and the only place in this file they
# are written down. Every modifier that draws an instruction paints it from
# here - the tape of the soup watcher, the ascii table of the extended machine,
# the program strip of the simple one - so a "<" is the same colour in every
# scene of the video, and the viewer can carry what a colour means from one
# shot to the next.
#
# The families are those of ``brainfuck/bff/bff_trace.py``: the two head moves
# share a colour and so do the two head0 moves, since which of a pair it is
# matters less than that it is a move at all; the two arithmetic instructions
# are deliberately *not* a pair, being the two a viewer has to tell apart most
# often; input and output go together, and so do the two brackets.
#
# The first element of each entry is the name of the ``Input Material`` node
# that carries the colour, so any of them can be swapped or animated through
# ``ibpy.get_geometry_node_from_modifier(modifier, "LessColor")``, and the
# ``colors=`` argument of every modifier here overrides one by that name.
INSTRUCTION_COLORS = (
    # node name,             colour,           character
    ("LessColor", "joker", "<"),
    ("MoreColor", "joker", ">"),
    ("CurlyBraceOpenColor", "custom1", "{"),
    ("CurlyBraceClosedColor", "custom1", "}"),
    ("PlusColor", "important", "+"),
    ("MinusColor", "orange", "-"),
    ("DotColor", "some_logo_blue", "."),
    ("CommaColor", "some_logo_blue", ","),
    ("BracketOpenColor", "x14_color", "["),
    ("BracketClosedColor", "x14_color", "]"),
)


def instruction_selector(tree, letter, colors, location=(0, 0), commands=None,
                         name="ColorSelector"):
    """One boolean per entry of *colors*: is *letter* the instruction it paints?

    The test every modifier in this file needs before it can paint a character
    from :data:`INSTRUCTION_COLORS`, in one node. ``in`` is ``Find in String``'s
    count of the character inside the set of that colour - note the order, set
    first and character second - so it is 1 exactly when the character is one
    of them, and a boolean socket reads any non-zero as true. Writing it this
    way round is what lets one formula serve an entry that covers a single
    character and one that covers a pair.

    The characters go into the formula in **single quotes**, and both of the
    jobs the quotes do matter: they keep ``<`` and ``>`` from being read as
    ``LESS_THAN`` and ``GREATER_THAN``, and they keep ``,`` from being read as
    the separator between two tokens - see ``split_rpn`` in
    ``geometry_nodes/nodes.py``.

    :param letter: string socket holding the character to test.
    :param colors: the palette, as ``(node name, colour, characters)`` triples.
    :param commands: optional string socket holding every character that is an
        instruction. Given one, the node gets an extra ``IsOperator`` output,
        true for a character that is any of them - what a caller needs to draw
        nothing at all for a byte that is not an instruction.
    :return: the group node, one boolean output per entry of *colors*, named
        after it.
    """
    labels = [node_name for node_name, _, _ in colors]
    functions = {node_name: "'%s',letter,in" % characters
                 for node_name, _, characters in colors}
    inputs = ["letter"]
    if commands is not None:
        functions["IsOperator"] = "commands,letter,in,0,>"
        inputs.append("commands")
        labels = labels + ["IsOperator"]

    selector = make_function(
        tree, name=name, location=location, hide=True,
        custom_ops={"in": {"type": FindInString, "inputs": ("String", "Search"),
                           "output": "Count", "label": "in"}},
        functions=functions, inputs=inputs, outputs=labels,
        strings=inputs, booleans=labels)
    tree.links.new(letter, selector.inputs["letter"])
    if commands is not None:
        tree.links.new(commands, selector.inputs["commands"])
    return selector


class BrainFuckSimpleModifier(GeometryNodesModifier):
    """
    A whole brainfuck machine running inside geometry nodes: a tape of cells
    whose values are incremented and decremented, a head that walks along it,
    and a program that is consumed one instruction per ``step_duration``.

    This is the python translation of the ``SimpleBrainFuck`` graph in
    ``video_bff/tmp.xml``, completed and debugged. Every frame of that graph is
    built by a private method of its own.

    **The encoding.** The machine does *not* use ascii for its output.
    ``code_table`` is ``"ABC...Z"``, so a cell holding 8 prints ``H`` and one
    holding 15 prints ``O``. That is what makes the program short enough to
    read: ``HELLO`` needs the values 8, 5, 12, 12, 15 rather than the ascii
    72, 69, 76, 76, 79. The table is drawn above the tape so the viewer can do
    the lookup along with the machine.

    **The state.** A simulation zone carries

    ``Geometry``
        the tape itself. This is the answer to "how do the cell values get
        incremented" - the values live in the integer point attribute
        ``Value`` of the tape geometry, and the zone hands that geometry from
        one frame to the next. ``+`` and ``-`` are a single ``Store Named
        Attribute`` whose *selection* is ``Index == PointerPosition``, so only
        the cell under the head changes; ``Sample Index`` reads that cell back
        out for ``.`` and for the increment itself.

    ``Counter``
        the program counter. The program itself is a *constant* - the machine
        reads its next instruction with ``Slice String(Program, Counter, 1)``
        and moves the counter rather than eating the string, which is what
        makes the loop instructions ``[`` and ``]`` possible: a jump is nothing
        but a different value written into ``Counter``. All seven instructions
        ``> < + - . [ ]`` are supported; ``,`` is not, there being nothing to
        read from.

    ``Output``, ``PointerPosition``, ``Step``, ``StartTime``, ``Time``
        what has been printed, the position of the head, the index of the
        current step, and the clock. The xml also carries ``Current``, the
        instruction being executed, for a read-out of its own; the program
        strip shows the same thing in place and it is gone.

    **The jumps.** Where a jump goes is not searched for at run time. The
    program is known when the graph is built, so :meth:`_jump_table` matches
    the brackets in python and the answer is baked into a second constant
    string, one character per instruction, holding the destination of a jump
    taken at that position offset by :attr:`JUMP_ORIGIN` to keep it printable.
    Looking it up is then the same ``Slice String`` and ``Char To Ascii`` pair
    that reads the instruction. The alternative - a repeat zone scanning for
    the matching bracket, counting depth - would put a search inside every
    frame of the simulation to compute something that cannot change.

    ``Time`` accumulates ``Delta Time`` rather than reading the scene clock, so
    the machine is driven by the simulation itself. Nothing happens before
    ``start_time``; after that the step index is
    ``floor((Time - StartTime) / StepDuration)`` and an instruction is executed
    on every frame where that index goes up *and* the program has not run out.
    The xml carries a second state item ``OldStep`` for the comparison, but it
    is written from the same socket as ``Step`` and so always holds the same
    value; ``Step`` alone is the previous step index, and that is all the
    comparison needs.

    Because the state lives in a simulation zone, the machine only runs when
    blender steps the scene forward one frame at a time. Jumping straight to a
    frame shows whatever the zone last cached, and ``render_with_skips`` will
    treat frames as still unless some *object* is animated - see the scene
    ``BffScene.simple_brain_fuck`` for what that means in practice.

    The frames of the graph:

    ``ControlParameter``, ``Variables``
        the constants - among them the program and its jump table - and the
        four state seeds (``Counter``, ``Output``, ``PointerPosition``,
        ``Step``). Both frames hand out what they hold as a single bundle.
        Sizes, positions, the two dozen materials and the program itself are
        wanted all over the graph, and as one wire each they came to
        fifty-seven lines crossing every frame between here and there. Now
        each frame downstream takes the one wire and opens it with a
        ``Separate Bundle`` naming just the entries it uses - :meth:`_unpack`
        - which is also the readable list of what that frame depends on.

    ``Tape``
        a ``Mesh Line`` of ``TapeSize`` points with ``Value = 0`` on every
        point - the tape as the machine starts it.

    ``RunProgram``
        the simulation zone: the clock, the instruction under the counter and
        the handover of every state item.

    ``Automaton``
        the instruction decoder. ``Char To Ascii`` turns the current
        instruction into its code, one ``Compare`` per opcode fires, and each
        comparison is ``AND``-ed with "a step has just begun" so that an
        instruction takes effect exactly once however many frames it is on
        screen for. It also works out where the counter goes next, which for
        five of the seven instructions is simply one on.

    Arithmetic that ran through a chain of ``Math`` nodes is written as a
    formula instead, in one :func:`~geometry_nodes.nodes.make_function` node
    per chain - the clock, the head, the jump condition, where a column of the
    strip stands. A chain of five nodes says what each step does and leaves
    the reader to work out what the whole is for; the formula says the whole
    and needs a comment for the steps. Its variables are named for what they
    are, so ``head,right,+,left,-,size,1,-,min,0,max`` reads as the sentence
    the head obeys. Careful with those names: a variable spelled like one of
    the operator tokens of the formula language is read as the *operator* -
    see the ``end`` of the clock below.

    ``Cells``, ``CellValues``
        the tape as it looks: one square per cell, coloured by whether the cell
        is still zero, holds a value, or is the one under the head, with the
        value written on it as a number by a *for each element* zone.

    ``CodeTable``, ``TableFrame``
        ``A``…1 to ``Z``…26 in a row above the tape, drawn by a repeat zone,
        with a rectangle around it that is sized from the bounding box of what
        came out - so it fits whatever alphabet is passed in.

    ``InputDisplay``, ``OutputDisplay``
        two framed boxes stacked below the tape: the whole program, and what
        has been printed so far. ``InputDisplay`` is as wide as the code table
        above the tape, so the three line up in one column.

    ``ProgramStrip``, ``CurrentDisplay``
        the program itself, written out once, one instruction per column across
        ``InputDisplay``, and the box that runs along it standing around the
        instruction the counter points at. The program does not move - the
        machine moves over it, which is how a program is normally read and
        makes a loop visible as the box running back and crossing the same
        instructions again. Each column is coloured first by which instruction
        it is - from :data:`INSTRUCTION_COLORS`, the palette shared by every
        modifier in this file - and then by what has become of it, so that what
        is dark behind the box is what has run for the last time and what is
        not is waiting for the next turn of its loop.

    ``SimulatedGeometry``
        the printed string in its box, the strip, and the head marker under the
        tape. Unlike in the xml this is built *outside* the simulation zone,
        from the state that the zone outputs: it is redrawn from scratch every
        frame, so carrying it through the zone as state would only feed it back
        to be thrown away.

    :param program: the brainfuck program, in ``> < + - . [ ]``
    :param code_table: the alphabet the cell values index into, 1-based
    :param tape_size: number of cells
    :param cell_size: width and height of a single cell
    :param step_duration: seconds one instruction is on screen. Must be at
        least two frames; below that the machine simply runs at one
        instruction per frame instead of skipping any.
    :param start_time: seconds before the first instruction runs
    :param tape_tilt: angle the tape is laid back by, so that a camera looking
        along +y sees the faces of the cells rather than their edge
    :param glyph_size: height of the number on a cell, as a fraction of
        ``CellSize``
    :param display_height: height of the two read-out boxes
    :param colors: optional ``{node name: colour name}`` overriding
        :attr:`CELL_COLORS`, :attr:`PROGRAM_COLORS`, any instruction of
        :data:`INSTRUCTION_COLORS`, and the two entries ``GlyphColor`` and
        ``FrameColor``
    """

    # "HELLO" as 8, 5, 12, 12, 15 - see the class docstring. Cell 0 is raised
    # to 8 and printed, the head steps right and cell 1 is raised to 5, then
    # the head steps back onto cell 0, which is topped up to 12 for the two
    # Ls and to 15 for the O. 27 instructions instead of the 70-odd an ascii
    # HELLO needs.
    HELLO = "++++++++.>+++++.<++++..+++."

    # The shortest HELLO there is with this alphabet, at 26 instructions. It
    # uses a single cell and never moves the head: 8 for the H, back down to 5
    # for the E, up to 12 for both Ls and to 15 for the O. Five "." are
    # unavoidable, and 8+3+7+0+3 = 21 is the shortest walk through 8, 5, 12,
    # 12, 15 - any second cell saves at most one "+" and costs two moves to
    # reach and come back from. See the class docstring.
    HELLO_SHORTEST = "++++++++.---.+++++++..+++."

    # The same output, with the 8 built by multiplication instead of by eight
    # "+". Each of these is one level deeper than the last: a loop, a loop
    # around a loop, and a loop around that. The tail is unchanged in all
    # three - only the cell the H ends up in moves one to the right per level,
    # because every level needs a counter cell of its own to count down.
    #
    # 4 x 2: cell 0 counts down from 4 and adds 2 to cell 1 each time round.
    HELLO_LOOP = "++++[>++<-]>.>+++++.<++++..+++."
    # 2 x (2 x 2): the outer loop runs the inner one twice, and the inner one
    # adds 2 to cell 2 twice - so cell 2 grows by 4 per turn of the outer loop.
    HELLO_LOOP2 = "++[>++[>++<-]<-]>>.>+++++.<++++..+++."
    # 2 x (2 x (2 x 1)): three counters, and the innermost loop adds a single 1
    # at a time. 107 steps to print HELLO, four times the straight version.
    HELLO_LOOP3 = "++[>++[>++[>+<-]<-]<-]>>>.>+++++.<++++..+++."

    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Colour of a cell by what is in it. The name is the name of the
    # ``Input Material`` node in the control frame, so any of them can be
    # swapped or animated through
    # ``ibpy.get_geometry_node_from_modifier(mod, "PointerColor")``.
    #
    # The chain is applied in this order and each link overrides the previous
    # one, so the first entry is the fall-back and the last one wins.
    CELL_COLORS = (
        ("ZeroColor", "gray_1"),  # nothing in it yet
        ("ValueColor", "drawing"),  # holds a value
        ("CurrentColor", "green"),  # the cell the head is on
    )

    POINTER_COLOR = "joker"
    GLYPH_COLOR = "text"  # the numbers on the cells and all the text
    FRAME_COLOR = "gray_2"  # the boxes around the displays and the code table

    # what an instruction of the program strip is painted, before anything has
    # happened to it: the shared palette, so that a "<" here is the "<" of
    # every other scene
    OPCODE_COLORS = INSTRUCTION_COLORS

    # ... and what becomes of that colour once the machine has been past.
    # Applied in this order, each overriding the last and all of them
    # overriding the instruction's own colour: "has run", then "has run but is
    # inside a loop that is still open, so it will run again". The instruction
    # being executed is painted last of all, in ``PointerColor``, the same
    # colour as the head marker, so that the two read as one thing.
    PROGRAM_COLORS = (
        ("DoneColor", "gray_2"),  # run, and not coming back
        ("WaitingColor", "example"),  # waiting for the next turn of its loop
    )

    # ascii codes of the seven instructions
    DOT, PLUS, MINUS, LEFT, RIGHT = ord("."), ord("+"), ord("-"), ord("<"), ord(">")
    OPEN, CLOSE = ord("["), ord("]")

    # The jump table is carried as a string, one character per instruction, so
    # that it can be read with the same Slice String the instruction is read
    # with. Destinations are offset by this so that they stay printable - the
    # lookup in Char To Ascii only covers codes 32 to 126, and 0 is a null
    # byte rather than a character in the first place.
    JUMP_ORIGIN = ord("0")

    # What a bundle item of each kind of control parameter is called. Blender
    # spells the type of a socket and the type of a bundle item the same way
    # everywhere except for floats, where the socket is a "VALUE" - and an
    # item of the wrong type is not an error, it is an item that silently goes
    # missing, so the two names are mapped here rather than written out at
    # every Separate Bundle. See :meth:`_unpack`.
    BUNDLE_TYPES = {"VALUE": "FLOAT", "INT": "INT", "VECTOR": "VECTOR",
                    "STRING": "STRING", "MATERIAL": "MATERIAL",
                    "BOOLEAN": "BOOLEAN", "RGBA": "RGBA"}

    # how the code table is laid out: one entry every ``table_spacing`` along
    # x, the letter ``table_line_gap`` below its number, and a frame around it
    # that is ``table_margin`` times the extent of the whole row
    table_spacing = 0.6
    table_line_gap = 0.7
    table_glyph_size = 0.5
    table_margin = 1.1
    frame_radius = 0.03

    # the program strip: one column per instruction, each column this much
    # wider than the letter that stands in it
    strip_glyph_size = 1.4
    # the box that runs along the strip marking the instruction about to be
    # executed - so many columns wide, and so much of the height of the display
    # it runs inside
    cursor_width = 1.3
    cursor_height = 0.7

    # the gap between the input display and the output display below it
    display_gap = 0.6

    def __init__(self, program=None, code_table=None, tape_size=5, cell_size=1,
                 step_duration=0.5, start_time=3.0, tape_tilt=0.4607669,
                 glyph_size=0.6, display_height=2.0, colors=None,
                 name="SimpleBrainFuck", **kwargs):
        self.program = self.HELLO if program is None else program
        self.jumps = self._encode_jumps(self.program)
        self.loops = self._encode_loop_starts(self.program)
        self.code_table = self.ALPHABET if code_table is None else code_table
        self.tape_size = tape_size
        self.cell_size = cell_size
        self.step_duration = step_duration
        self.start_time = start_time
        self.tape_tilt = tape_tilt
        self.glyph_size = glyph_size
        self.display_height = display_height
        overrides = colors or {}
        self.cell_colors = tuple((node_name, overrides.get(node_name, color))
                                 for node_name, color in self.CELL_COLORS)
        self.opcode_colors = tuple(
            (node_name, overrides.get(node_name, color), characters)
            for node_name, color, characters in self.OPCODE_COLORS)
        self.program_colors = tuple((node_name, overrides.get(node_name, color))
                                    for node_name, color in self.PROGRAM_COLORS)
        self.glyph_color = overrides.get("GlyphColor", self.GLYPH_COLOR)
        self.frame_color = overrides.get("FrameColor", self.FRAME_COLOR)
        self.kwargs = kwargs
        super().__init__(name=name, automatic_layout=False)

    # ----------------------------------------------------------------
    @classmethod
    def _jump_table(cls, program):
        """Where a jump taken at each position of *program* goes.

        ``[`` on a zero cell continues after its matching ``]``, and ``]`` on a
        cell that is not zero goes back to just after its matching ``[`` - so
        the body of the loop runs again without the ``[`` being re-read.
        Positions that are not brackets never jump and hold 0.

        :return: a list of destinations, one per instruction.
        :raises ValueError: if the brackets do not match.
        """
        targets, open_at = [0] * len(program), []
        for position, instruction in enumerate(program):
            if instruction == "[":
                open_at.append(position)
            elif instruction == "]":
                if not open_at:
                    raise ValueError("']' without a matching '[' at position "
                                     "%d of %r" % (position, program))
                start = open_at.pop()
                targets[start] = position + 1
                targets[position] = start + 1
        if open_at:
            raise ValueError("'[' without a matching ']' at position %d of %r"
                             % (open_at[-1], program))
        return targets

    @classmethod
    def _encode_jumps(cls, program):
        """The jump table of *program* as the string the graph looks up in.

        :return: one character per instruction, its code the destination of a
            jump taken there plus :attr:`JUMP_ORIGIN`.
        :raises ValueError: if the brackets do not match, or the program is too
            long for its destinations to stay inside the printable ascii range.
        """
        reach = CharToAscii.LAST_PRINTABLE - cls.JUMP_ORIGIN
        if len(program) > reach:
            raise ValueError("a program of more than %d instructions cannot "
                             "have its jump table encoded as printable "
                             "characters; this one has %d"
                             % (reach, len(program)))
        return "".join(chr(cls.JUMP_ORIGIN + target)
                       for target in cls._jump_table(program))

    @classmethod
    def _loop_starts(cls, program):
        """For each position, the outermost ``[`` that is still open there.

        Answers "if the machine is here, which instructions are going to run
        again?" - everything from that bracket up to the current one. The
        *outermost* open bracket rather than the innermost, because an
        instruction in the body of an outer loop is just as much waiting for
        its next turn as one in the inner loop that is running now.

        Both brackets count as inside their own loop, so standing on the ``]``
        marks the whole body, and standing on the ``[`` marks nothing - which
        is right, since at that point nothing of this loop has run yet.

        :return: a list of ``[`` positions, 1-based so that 0 can mean "not
            inside a loop at all".
        """
        starts, open_at = [0] * len(program), []
        for position, instruction in enumerate(program):
            if instruction == "[":
                open_at.append(position)
            starts[position] = open_at[0] + 1 if open_at else 0
            if instruction == "]":
                open_at.pop()
        return starts

    @classmethod
    def _encode_loop_starts(cls, program):
        """The loop-start table of *program*, encoded like :meth:`_encode_jumps`.

        :return: one character per instruction, its code the 1-based position
            of the outermost open ``[`` plus :attr:`JUMP_ORIGIN`.
        """
        return "".join(chr(cls.JUMP_ORIGIN + start)
                       for start in cls._loop_starts(program))

    @classmethod
    def simulate(cls, program, tape_size=5, code_table=None):
        """Run *program* in python exactly as the graph runs it.

        The animation is as long as the machine takes, and that is the number
        of instructions *executed*, not the length of the program - a loop of
        four turns is one instruction on screen at a time. So the scene needs
        to be able to ask. It is also what the graph is verified against.

        :return: ``(steps, output, tape)``.
        """
        table = cls.ALPHABET if code_table is None else code_table
        jumps = cls._jump_table(program)
        tape, head, counter, steps, output = [0] * tape_size, 0, 0, 0, ""
        while counter < len(program):
            instruction, cell = program[counter], tape[head]
            onward = counter + 1
            if instruction == ">":
                head = min(head + 1, tape_size - 1)
            elif instruction == "<":
                head = max(head - 1, 0)
            elif instruction == "+":
                tape[head] = cell + 1
            elif instruction == "-":
                tape[head] = cell - 1
            elif instruction == "." and cell > 0:
                output += table[cell - 1]
            elif instruction == "[" and cell == 0:
                onward = jumps[counter]
            elif instruction == "]" and cell != 0:
                onward = jumps[counter]
            counter, steps = onward, steps + 1
        return steps, output, tape

    # ----------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        control = self._create_control_frame(tree)
        variables = self._create_variables_frame(tree)
        tape = self._create_tape_frame(tree, control)
        run = self._create_run_program_frame(tree, control, variables, tape)

        cells = self._create_cells_frame(tree, control, variables, run)
        table = self._create_code_table_frame(tree, control)
        displays = [
            self._create_display_frame(tree, control, "InputDisplay",
                                       "InputDisplaySize", "InputPosition",
                                       location=(26, -21), control_at=(27, -19.0)),
            self._create_display_frame(tree, control, "OutputDisplay",
                                       "OutputDisplaySize", "OutputPosition",
                                       location=(26, -24), control_at=(30, -24)),
        ]
        simulated = self._create_simulated_geometry_frame(tree, control, variables, run)

        out = self.group_outputs
        out.location = (38 * 200, -2 * 200)
        join = JoinGeometry(tree, location=(36, -4))
        extra = self._extra_geometry(tree, control, variables, run)
        for piece in [cells, table, simulated] + displays + extra:
            tree.links.new(piece, join.geometry_in)
        tree.links.new(join.geometry_out, out.inputs["Geometry"])

    # ----------------------------------------------------------------
    def _create_control_frame(self, tree):
        """``ControlParameter``: every constant of the machine.

        The parameters leave the frame as a single bundle rather than as one
        wire each - see the :meth:`_bundle` at the end of the method, and
        :meth:`_unpack`, which is how every frame downstream gets at them.

        :return: ``{name: node}``, so that the frames downstream can pick the
            parameter they need by the name it carries in the editor, plus
            ``Bundle``, the one node that carries all of them at once.
        """
        x = -23.8
        control = {
            "TapeSize": InputInteger(tree, location=(x, 0), integer=self.tape_size,
                                     name="TapeSize"),
            "CellSize": InputValue(tree, location=(x, -0.8), value=self.cell_size,
                                   name="CellSize"),
            "StartTime": InputValue(tree, location=(x, -1.6), value=self.start_time,
                                    name="StartTime"),
            "StepDuration": InputValue(tree, location=(x, -2.4), value=self.step_duration,
                                       name="StepDuration"),
            "CodeTable": InputString(tree, location=(x, -3.2), string=self.code_table,
                                     name="CodeTable"),
        }

        # one Input Material node per colour of a cell, then one per
        # instruction, then what has become of an instruction, then the two
        # that everything else is drawn in
        palette = {}
        rows = ([(node_name, color) for node_name, color in self.cell_colors]
                + [(node_name, color) for node_name, color, _ in self.opcode_colors])
        for row, (node_name, color) in enumerate(rows):
            palette[node_name] = InputMaterial(tree, location=(x, -4.4 - 0.4 * row),
                                               material=color, name=node_name,
                                               **self.kwargs)
        rest = list(self.program_colors) + [("PointerColor", self.POINTER_COLOR),
                                            ("GlyphColor", self.glyph_color),
                                            ("FrameColor", self.frame_color)]
        for offset, (node_name, color) in enumerate(rest):
            palette[node_name] = InputMaterial(
                tree, location=(x, -4.4 - 0.4 * (len(rows) + offset)),
                material=color, name=node_name, hide=True, **self.kwargs)
        for source in palette.values():
            self.materials.append(source.node.material)
        control.update(palette)

        # The read-outs below the tape and the code table above it are all
        # centred on the middle of the tape, which runs from x=0 to
        # x=TapeSize*CellSize. Everything the machine shows is therefore
        # stacked in one column and a camera looking along +y frames it whole.
        #
        # The input display holds the whole program at once, so it is made as
        # wide as the code table above it and the two line up. That width is
        # taken from the table's layout rather than measured off the geometry:
        # the table is only sized from its bounding box because it has to fit
        # whatever letters are in it, and a display that changed width with the
        # alphabet would be worse, not better.
        middle = 0.5 * self.tape_size * self.cell_size
        table_width = ((len(self.code_table) - 1) * self.table_spacing
                       * self.table_margin)
        for row, (node_name, value) in enumerate((
                ("InputDisplaySize", table_width), ("OutputDisplaySize", 6.0))):
            control[node_name] = InputValue(tree, location=(x, -7.0 - 0.4 * row),
                                            value=value, name=node_name)
        # the output sits underneath the input rather than beside it - there is
        # no room beside a display that spans the whole width
        below = -3.0 - self.display_height - self.display_gap
        for row, (node_name, value) in enumerate((
                ("InputPosition", [middle, 0, -3.0]),
                ("OutputPosition", [middle, 0, below]),
                ("TapePosition", [0, 0, 0]))):
            control[node_name] = InputVector(tree, location=(x, -8.6 - 0.8 * row),
                                             vector=Vector(value), name=node_name)
        # where the code table starts - its entries grow to the right from
        # here, so it is shifted left by half its own width - and how far the
        # head marker hangs below the tape
        table_start = middle - 0.5 * (len(self.code_table) - 1) * self.table_spacing
        control["TablePosition"] = InputVector(tree, location=(x, -11.0),
                                               vector=Vector([table_start, 0, 2.8]),
                                               name="TablePosition")
        control["PointerOffset"] = InputVector(tree, location=(x, -11.8),
                                               vector=Vector([0, 0, -0.9 * self.cell_size]),
                                               name="PointerOffset")

        # Everything above leaves this frame as one bundle. It used to leave
        # as forty-seven separate wires, every one of them running most of the
        # width of the graph and crossing everything in between - ten of them
        # for the colours of the instructions alone. Now each frame takes the
        # one wire and says in a Separate Bundle of its own what it is
        # unpacking, which doubles as a list of what that frame depends on;
        # see :meth:`_bundle` and :meth:`_unpack`.
        #
        # The Input nodes stay where they are: the bundle gathers them rather
        # than replacing them, so every parameter is still a node of its own
        # to find by name with ``ibpy.get_geometry_node_from_modifier`` and to
        # keyframe. Adding one here is enough to make it available everywhere.
        # anything a subclass wants handed out with the rest, before the
        # bundle is tied - see BrainFuckTransitionModifier._more_control
        self._more_control(tree, control, x)
        self._bundle(tree, control, location=(x + 1.6, 0), name="Control")

        for node in control.values():
            node.hide = True

        frame = Frame(tree, location=(-24, 0.6), label="ControlParameter")
        frame.add(list(control.values()))
        return control

    # ----------------------------------------------------------------
    def _bundle(self, tree, group, location=(0, 0), name="Bundle"):
        """Tie everything in *group* into one bundle, and put that in there too.

        :param group: the ``{name: node}`` of a frame that has to hand things
            to the rest of the graph, as :meth:`_create_control_frame` and
            :meth:`_create_variables_frame` build them. The bundle joins it
            under ``Bundle``, which is where :meth:`_unpack` looks for it.
        :return: the ``Combine Bundle`` node.
        """
        # the items are read off before the bundle joins the group, or it
        # would be asked to carry itself
        items = [(node_name, self.BUNDLE_TYPES[node.std_out.type], node.std_out)
                 for node_name, node in group.items()]
        group["Bundle"] = CombineBundle(tree, location=location, name=name,
                                        items=items)
        return group["Bundle"]

    # ----------------------------------------------------------------
    def _unpack(self, tree, group, *names, location=(0, 0), name="BundleIn"):
        """The entries *names* of *group*, taken back out of its bundle.

        What a frame calls instead of reaching across the graph for each thing
        it needs. The types are read off the nodes the bundle was built from
        rather than written out here, since a Separate Bundle that asks for
        the wrong type does not fail - it hands out a default and the graph
        goes quietly wrong.

        :return: the ``Separate Bundle`` node. Its sockets come out by name:
            ``dials.out("CellSize")``.
        """
        items = [(entry, self.BUNDLE_TYPES[group[entry].std_out.type])
                 for entry in names]
        return SeparateBundle(tree, location=location, name=name,
                              bundle=group["Bundle"].std_out, items=items)

    # ----------------------------------------------------------------
    def _more_control(self, tree, control, x):
        """Control parameters of a subclass, added before the bundle is tied.

        Nothing here. A subclass puts its own ``Input`` nodes into *control*
        and they travel with the others - see
        :meth:`BrainFuckTransitionModifier._more_control`.

        :param x: the column the control frame is written in.
        """
        return

    # ----------------------------------------------------------------
    def _tape_stand(self, tree, control, position, location=(0, 0),
                    name="TapeStand"):
        """Where the tape stands, and the markers that point at it with it.

        ``TapePosition`` itself, here. :class:`BrainFuckTransitionModifier`
        adds a drift to it, which is how the whole tape steps down out of the
        way of the ascii table coming in.

        :return: the vector socket to place the tape by, and the nodes that
            made it - none here.
        """
        return position, []

    # ----------------------------------------------------------------
    def _output_string(self, tree, control, text):
        """What the output display reads. What the machine printed, here.

        :class:`BrainFuckTransitionModifier` empties it once the same string
        has arrived on the tape, which is what leaves the display an empty
        frame for the morph.
        """
        return text

    # ----------------------------------------------------------------
    def _extra_geometry(self, tree, control, variables, run):
        """Whatever else a subclass draws, to be joined with the machine.

        :return: a list of geometry sockets - empty here.
        """
        return []

    # ----------------------------------------------------------------
    def _tape_in_zone(self, tree, control, sim_in):
        """The tape the automaton reads and writes, inside the simulation zone.

        Here it is simply the zone's own state: the tape is seeded once from
        the ``Tape`` frame and handed from one frame to the next, which is
        what keeps the cell values alive - and what fixes the shape of the
        tape at the frame the simulation started.
        :class:`BrainFuckTransitionModifier` rebuilds it instead.

        :return: the geometry socket, and the nodes that made it - none here.
        """
        return sim_in.outputs["Geometry"], []

    # ----------------------------------------------------------------
    def _create_variables_frame(self, tree):
        """``Variables``: the program, its jump table and the four state seeds.

        These leave the frame as one bundle too - see
        :meth:`_create_control_frame` for why, and :meth:`_unpack` for how
        they are taken out again.

        :return: ``{name: node}`` plus ``Bundle``, the node that carries all
            of them at once.
        """
        x = -15.8
        variables = {
            # the program is a constant, not a state item: the machine walks a
            # counter along it instead of eating it, which is what lets a loop
            # jump backwards into a part it has already run
            "Input": InputString(tree, location=(x, 0), string=self.program,
                                 name="Program", label="Input"),
            # where a jump taken at each position goes, offset into the
            # printable range - see _encode_jumps
            "Jumps": InputString(tree, location=(x, -0.8), string=self.jumps,
                                 name="JumpTable", label="Jumps"),
            # which loop is open at each position, encoded the same way. Only
            # the program strip needs it - the machine itself does not care.
            "Loops": InputString(tree, location=(x, -1.6), string=self.loops,
                                 name="LoopTable", label="Loops"),
            "Output": InputString(tree, location=(x, -2.4), string="",
                                  name="OutputStart", label="Output"),
            "Pointer": InputInteger(tree, location=(x, -3.2), integer=0,
                                    name="PointerPosition"),
            "Counter": InputInteger(tree, location=(x, -4.0), integer=0,
                                    name="ProgramCounter"),
            # -1, so that the first step (index 0) counts as an advance and the
            # first instruction is executed rather than skipped
            "Step": InputInteger(tree, location=(x, -4.8), integer=-1, name="Step"),
        }
        # None of these is read where it stands. The program is read twice by
        # the machine and twice more by the strip that draws it, away at the
        # right of the graph; the jump table only in the automaton, the loop
        # table only in the strip, and the four seeds by the simulation zone
        # - ten wires, and every one of them a long one.
        self._bundle(tree, variables, location=(x + 1.6, 0), name="Variables")

        frame = Frame(tree, location=(-16, 0.6), label="Variables")
        frame.add(list(variables.values()))
        return variables

    # ----------------------------------------------------------------
    def _create_tape_frame(self, tree, control):
        """``Tape``: the cells as the machine starts them, all holding zero.

        :return: the geometry socket of the initial tape.
        """
        dials = self._unpack(tree, control, "TapeSize", "CellSize",
                             location=(-8.6, 0.6), name="TapeControl")
        end = make_function(tree, name="EndPosition",
                            functions={
                                "end": "e_x,tapeSize,cellSize,*,scale"
                            }, inputs=["tapePosition", "tapeSize", "cellSize"], outputs=["end"],
                            scalars=["cellSize"], integers=["tapeSize"], vectors=["end"], hide=True, location=(-7, 0))
        tree.links.new(dials.out("TapeSize"), end.inputs["tapeSize"])
        tree.links.new(dials.out("CellSize"), end.inputs["cellSize"])

        line = MeshLine(tree, location=(-6, 0.6), mode="END_POINTS",
                        count=dials.out("TapeSize"),
                        start_location=Vector(), end_location=end.outputs["end"])
        # every cell starts empty. The attribute has to exist from the first
        # frame on, otherwise the "Sample Index" in the automaton has nothing
        # to read and the cells have nothing to be coloured by.
        zeros = StoredNamedAttribute(tree, location=(-4.6, 0.6), data_type="INT",
                                     domain="POINT", name="Value", value=0,
                                     label="ClearTape")
        create_geometry_line(tree, [line, zeros])
        frame = Frame(tree, location=(-8.2, 1.4), label="Tape")
        frame.add([dials, end, line, zeros])
        return zeros.geometry_out

    # ----------------------------------------------------------------
    def _create_run_program_frame(self, tree, control, variables, tape):
        """``RunProgram``: the simulation zone - the clock and the program counter.

        :return: ``{name: socket}`` of the state as it leaves the zone.
        """
        dials = self._unpack(tree, control, "StartTime", "StepDuration",
                             location=(0.4, 6.4), name="RunControl")
        source = self._unpack(tree, variables, "Input", "Output", "Pointer",
                              "Counter", "Step", location=(0.4, 4.4),
                              name="RunVariables")
        zone = Simulation(tree, location=(2, 5), node_width=20, geometry=tape)
        sim_in, sim_out = zone.simulation_input, zone.simulation_output
        for socket_type, socket_name, initial in (
                ("FLOAT", "StartTime", dials.out("StartTime")),
                ("INT", "Step", source.out("Step")),
                ("INT", "PointerPosition", source.out("Pointer")),
                ("INT", "Counter", source.out("Counter")),
                ("STRING", "Output", source.out("Output")),
                ("FLOAT", "Time", 0.0)):
            zone.add_socket(socket_type=socket_type, name=socket_name, value=initial)

        # --- the instruction under the counter --------------------------
        program = source.out("Input")
        current = SliceString(tree, location=(3.2, 4.6), string=program,
                              position=sim_in.outputs["Counter"], length=1,
                              name="Instruction")
        opcode = CharToAscii(tree, location=(4.4, 4.6), char=current.std_out)
        length = StringLength(tree, location=(3.2, 3.6), string=program,
                              name="ProgramLength")
        # --- the clock ---------------------------------------------------
        # The whole of the machine's sense of time, in one node:
        #
        # ``clock``
        #     the zone's own Delta Time added to what it carried in, rather
        #     than the scene clock - so the machine keeps its own time and a
        #     state item is all that is needed.
        # ``index``
        #     which step the clock has reached. ``max`` at -1 keeps it there
        #     while the machine is still waiting, so that the first real step
        #     (index 0) is an increase and fires the first instruction.
        #     ``floor``, not truncation: truncation rounds towards zero, so
        #     the whole last StepDuration before StartTime would already come
        #     out as step 0 and fire the first instruction early.
        # ``Fire``
        #     an instruction is executed on the one frame where the step index
        #     goes up, never on the frames in between - otherwise a single "+"
        #     would count once per rendered frame. Comparing against the
        #     *previous* index rather than using the difference as a count
        #     also means that a step_duration shorter than a frame degrades to
        #     one instruction per frame instead of skipping instructions.
        #
        #     ...and the counter has to be still inside the program. The clock
        #     keeps going after the last instruction, so without that the
        #     machine would go on "executing" the empty slice past the end of
        #     it: the head would stay put but the read-out would blank and the
        #     tape would keep being rewritten. Halting when the counter runs
        #     off the end leaves the finished state up.
        # ``end`` rather than ``length`` for the length of the program: a
        # variable named after one of make_function's own operator tokens
        # (see OPERATORS in ibpy) is read as the operator, and ``length`` is
        # the length of a *vector* - which builds without complaint and
        # compares the wrong thing.
        clock = make_function(
            tree, name="Clock", location=(6.4, 6.4), hide=False,
            aux_functions={"clock": "delta,time,+",
                           "index": "clock,start,-,duration,/,-1,max,floor"},
            functions={"Time": "clock", "Step": "index",
                       "Fire": "index,old,>,counter,end,<,and"},
            inputs=["delta", "time", "start", "duration", "old", "counter",
                    "end"],
            outputs=["Time", "Step", "Fire"],
            scalars=["delta", "time", "start", "duration", "Time",
                     "clock", "index"],
            integers=["old", "counter", "end", "Step"],
            booleans=["Fire"])
        for socket, socket_name in (
                (sim_in.outputs["Delta Time"], "delta"),
                (sim_in.outputs["Time"], "time"),
                (sim_in.outputs["StartTime"], "start"),
                (dials.out("StepDuration"), "duration"),
                (sim_in.outputs["Step"], "old"),
                (sim_in.outputs["Counter"], "counter"),
                (length.std_out, "end")):
            tree.links.new(socket, clock.inputs[socket_name])
        time, step, fire = (clock.outputs["Time"], clock.outputs["Step"],
                            clock.outputs["Fire"])

        # --- the reroutes that carry the decoded step into the automaton ---
        code_in = Reroute(tree, location=(11.6, 4.6), ins=opcode.std_out, name="Opcode")
        fire_in = Reroute(tree, location=(11.6, 4.2), ins=fire, name="Fire")
        head_in = Reroute(tree, location=(11.6, 3.8),
                          ins=sim_in.outputs["PointerPosition"], name="Head")
        step_in = Reroute(tree, location=(11.6, 3.4), ins=sim_in.outputs["Counter"],
                          name="Counter")

        tape_in, rebuilt = self._tape_in_zone(tree, control, sim_in)
        pointer, tape_out, output, counter = self._create_automaton_frame(
            tree, control, variables, sim_in, code_in.std_out, fire_in.std_out,
            head_in.std_out, step_in.std_out, tape_in)

        for socket, name in ((time, "Time"), (step, "Step"),
                             (sim_in.outputs["StartTime"], "StartTime"),
                             (counter, "Counter"),
                             (pointer, "PointerPosition"), (output, "Output")):
            tree.links.new(socket, sim_out.inputs[name])
        # replaces the pass-through that the Simulation wrapper puts in
        tree.links.new(tape_out, sim_out.inputs["Geometry"])

        frame = Frame(tree, location=(1.6, 7.4), label="RunProgram")
        frame.add([dials, source, zone, clock, current, opcode, length,
                   code_in, fire_in, head_in, step_in] + rebuilt)
        return {name: sim_out.outputs[name] for name in
                ("Geometry", "Step", "PointerPosition", "Counter", "Output")}

    # ----------------------------------------------------------------
    def _create_automaton_frame(self, tree, control, variables, sim_in, opcode,
                                fire, head, counter, tape_in):
        """``Automaton``: what the seven instructions do.

        Every instruction is one ``Compare`` against its ascii code, ``AND``-ed
        with *fire* - "a new step has just begun". Without that ``AND`` an
        instruction would take effect once per rendered frame instead of once.
        Four of them are decoded by :func:`decodes` into a node of their own,
        because what they do with the answer is a node too; the other three -
        ".", "[" and "]" - are decoded inside the formula that acts on them,
        which is where their ``AND`` with *fire* has gone.

        :param counter: the program counter as the frame starts
        :param tape_in: the tape to read the cell values from and write them
            back to - see :meth:`_tape_in_zone`
        :return: ``(pointer, tape, output, counter)`` sockets for the state to
            be written back into the simulation.
        """
        dials = self._unpack(tree, control, "TapeSize", "CodeTable",
                             location=(12.4, -5.4), name="AutomatonControl")
        source = self._unpack(tree, variables, "Jumps", location=(12.4, -6.8),
                              name="AutomatonVariables")
        built = [dials, source]

        def keep(node):
            built.append(node)
            return node

        def decodes(code, row, label):
            """``True`` on the frame the instruction ``code`` is executed."""
            is_op = keep(CompareNode(tree, location=(12.4, row), operation="EQUAL",
                                     data_type="INT", inputs0=opcode, inputs1=code,
                                     name="Is" + label, hide=True))
            return keep(BooleanMath(tree, location=(13.6, row), operation="AND",
                                    inputs0=is_op.std_out, inputs1=fire,
                                    name="Do" + label, hide=True)).std_out

        def step_of(condition, row, label):
            """1 on the frame the instruction runs, 0 otherwise."""
            return keep(Switch(tree, location=(14.8, row), input_type="INT",
                               switch=condition, false=0, true=1, name=label)).std_out

        # --- the head --------------------------------------------------
        right = step_of(decodes(self.RIGHT, 3.2, "Right"), 3.2, "StepRight")
        left = step_of(decodes(self.LEFT, 2.4, "Left"), 2.4, "StepLeft")
        # one step right, one step left, and the tape does not wrap and does
        # not grow - so the head is kept on it. Without the min and the max a
        # program with one ">" too many would write into a cell that is not
        # drawn and silently do nothing visible.
        pointer = keep(make_function(
            tree, name="MoveHead", location=(17.0, 2.8), hide=False,
            functions={"Head": "head,right,+,left,-,size,1,-,min,0,max"},
            inputs=["head", "right", "left", "size"], outputs=["Head"],
            integers=["head", "right", "left", "size", "Head"]))
        for socket, socket_name in ((head, "head"), (right, "right"),
                                    (left, "left"),
                                    (dials.out("TapeSize"), "size")):
            tree.links.new(socket, pointer.inputs[socket_name])

        # --- the cell under the head ------------------------------------
        # this is where the values live: an integer attribute of the tape
        # geometry, which the simulation zone hands from frame to frame
        stored = NamedAttribute(tree, location=(12.4, 0.2), data_type="INT",
                                name="Value")
        cell = SampleIndex(tree, location=(13.6, 0.2), data_type="INT", domain="POINT",
                           geometry=tape_in, value=stored.std_out,
                           index=head, name="CellUnderHead")
        plus = step_of(decodes(self.PLUS, 1.2, "Plus"), 1.2, "Increment")
        minus = step_of(decodes(self.MINUS, 0.4, "Minus"), 0.4, "Decrement")
        raised = IntegerMath(tree, location=(16.0, 0.8), operation="ADD",
                             inputs0=cell.std_out, inputs1=plus, name="CellPlus")
        lowered = IntegerMath(tree, location=(17.0, 0.8), operation="SUBTRACT",
                              inputs0=raised.std_out, inputs1=minus, name="CellMinus")
        # only the cell the head is on is written, every other one keeps what
        # it had - this selection is the whole of "+" and "-"
        here = Index(tree, location=(16.0, -0.6))
        selection = CompareNode(tree, location=(17.0, -0.6), operation="EQUAL",
                                data_type="INT", inputs0=here.std_out, inputs1=head,
                                name="AtTheHead", hide=True)
        tape = StoredNamedAttribute(tree, location=(18.4, 0.2), data_type="INT",
                                    domain="POINT", name="Value",
                                    selection=selection.std_out, value=lowered.std_out,
                                    label="WriteCell")
        tree.links.new(tape_in, tape.geometry_in)

        # --- printing ---------------------------------------------------
        # the point of the exercise: the cell value indexes into the code
        # table, so 8 prints H. The table is 1-based, hence the -1.
        # ``Index`` is where in the table to look, 1-based, hence the -1.
        # ``Print`` is whether to look at all: only on the frame a "." is
        # executed, and only if the cell holds something. Without the second
        # half a "." on a zero cell would print an A, because slicing at -1
        # clamps to the front of the table.
        prints = make_function(
            tree, name="Print", location=(13.6, 5.6), hide=False,
            functions={"Print": "opcode,%d,=,fire,and,cell,0,>,and" % self.DOT,
                       "Index": "cell,1,-"},
            inputs=["opcode", "fire", "cell"], outputs=["Print", "Index"],
            integers=["opcode", "cell", "Index"], booleans=["fire", "Print"])
        for socket, socket_name in ((opcode, "opcode"), (fire, "fire"),
                                    (cell.std_out, "cell")):
            tree.links.new(socket, prints.inputs[socket_name])
        letter = SliceString(tree, location=(14.8, 4.8), string=dials.out("CodeTable"),
                             position=prints.outputs["Index"], length=1,
                             name="Letter")
        printed = Switch(tree, location=(16.0, 5.6), input_type="STRING",
                         switch=prints.outputs["Print"], false="",
                         true=letter.std_out, name="Printed")
        # The empty string of a step that does not print appends nothing, so
        # the join can run unconditionally. The order of the two links matters
        # and is the reverse of the order they are made in: blender puts the
        # newest link into a multi-input socket on top, and Join Strings
        # concatenates top to bottom. Linking what has been printed so far
        # *second* is what makes the output read HELLO rather than OLLEH.
        output = StringJoin(tree, location=(17.4, 5.6), delimiter="",
                            strings=printed.std_out, name="Print")
        tree.links.new(sim_in.outputs["Output"], output.node.inputs["Strings"])

        # --- the loop, and where the counter goes next -------------------
        # "[" and "]" are the same instruction read in opposite directions:
        # each looks at the cell under the head and either falls through to the
        # next instruction or jumps to the destination the table holds for it.
        # "[" leaves the loop when the cell has run down to zero, "]" goes
        # round again while it has not.
        jumping = make_function(
            tree, name="TakeJump", location=(15.0, -2.0), hide=False,
            aux_functions={"open": "opcode,%d,=,fire,and" % self.OPEN,
                           "close": "opcode,%d,=,fire,and" % self.CLOSE,
                           "empty": "cell,0,="},
            functions={"Jump": "open,empty,and,close,empty,not,and,or"},
            inputs=["opcode", "fire", "cell"], outputs=["Jump"],
            integers=["opcode", "cell"], booleans=["fire", "Jump"],
            scalars=["open", "close", "empty"])
        for socket, socket_name in ((opcode, "opcode"), (fire, "fire"),
                                    (cell.std_out, "cell")):
            tree.links.new(socket, jumping.inputs[socket_name])

        # the destination is not searched for: it was worked out in python when
        # the graph was built and baked into a string with one character per
        # instruction, so reading it is the same slice-and-decode the
        # instruction itself goes through
        entry = SliceString(tree, location=(12.4, -3.4),
                            string=source.out("Jumps"), position=counter,
                            length=1, name="JumpEntry")
        encoded = CharToAscii(tree, location=(13.6, -3.4), char=entry.std_out,
                              name="JumpCode")
        target = IntegerMath(tree, location=(14.8, -3.4), operation="SUBTRACT",
                             inputs0=encoded.std_out, inputs1=self.JUMP_ORIGIN,
                             name="JumpTarget")
        onward = IntegerMath(tree, location=(14.8, -4.2), operation="ADD",
                             inputs0=counter, inputs1=1, name="NextInstruction")
        jumped = Switch(tree, location=(18.0, -3.4), input_type="INT",
                        switch=jumping.outputs["Jump"], false=onward.std_out,
                        true=target.std_out, name="CounterAfterStep")
        # on the frames in between two steps, and after the program has ended,
        # the counter stays where it is
        moved = Switch(tree, location=(19.0, -3.4), input_type="INT",
                       switch=fire, false=counter, true=jumped.std_out,
                       name="NewCounter")

        frame = Frame(tree, location=(12.0, 7.0), label="Automaton")
        frame.add(built + [stored, cell, raised, lowered, here, selection, tape,
                           prints, letter, printed, output, jumping,
                           entry, encoded, target, onward, jumped, moved])
        return (pointer.outputs["Head"], tape.geometry_out, output.std_out,
                moved.std_out)

    # ----------------------------------------------------------------
    def _create_cells_frame(self, tree, control, variables, run):
        """``Cells``: the tape as it looks, coloured by what is in it.

        A filled square is instanced onto every tape point and the instances
        are realized, so that the point attribute ``Value`` reaches the faces.
        A chain of ``Set Material`` then paints them: the first link has no
        selection and is the fall-back, each later one overrides it where its
        selection holds. The number in the cell is built by a *for each
        element* zone, because ``Value to String`` needs a single value and a
        string is not a field.

        :return: the geometry socket of the finished tape.
        """
        tape = run["Geometry"]
        dials = self._unpack(tree, control, "CellSize", "TapePosition",
                             *[node_name for node_name, _ in self.cell_colors],
                             location=(25, 2.6), name="CellsControl")
        quad = Quadrilateral(tree, location=(26, 2), mode="RECTANGLE",
                             width=dials.out("CellSize"),
                             height=dials.out("CellSize"))
        fill = FillCurve(tree, location=(27, 2), mode="N-gons")
        create_geometry_line(tree, [quad, fill])
        instances = InstanceOnPoints(tree, location=(28, 2.6), points=tape,
                                     instance=fill.geometry_out)
        realize = RealizeInstances(tree, location=(29, 2.6))

        value = NamedAttribute(tree, location=(28, 1.2), data_type="INT", name="Value")
        here = Index(tree, location=(28, 0.6))
        holds = CompareNode(tree, location=(29, 1.2), operation="NOT_EQUAL",
                            data_type="INT", inputs0=value.std_out, inputs1=0,
                            name="CellHoldsAValue", hide=True)
        under = CompareNode(tree, location=(29, 0.6), operation="EQUAL",
                            data_type="INT", inputs0=here.std_out,
                            inputs1=run["PointerPosition"], name="CellUnderHead",
                            hide=True)
        selections = (None, holds.std_out, under.std_out)

        painters = [SetMaterial(tree, location=(30 + column, 2.6), selection=selection,
                                material=dials.out(node_name),
                                name="Paint" + node_name)
                    for column, ((node_name, _), selection)
                    in enumerate(zip(self.cell_colors, selections))]
        create_geometry_line(tree, [instances, realize] + painters)

        numbers = self._create_cell_values(tree, control, variables, run)
        joined = JoinGeometry(tree, location=(34, 2.6))
        tree.links.new(painters[-1].geometry_out, joined.geometry_in)
        tree.links.new(numbers, joined.geometry_in)
        # the tape lies in the x-y plane, which a camera looking along +y sees
        # edge-on. Laying it back brings the faces of the cells into view; the
        # numbers are pre-turned by the complement of this angle in
        # _create_cell_values, so that they come out upright.
        stand, drift = self._tape_stand(tree, control, dials.out("TapePosition"),
                                        location=(34, 3.4), name="CellsStand")
        tilt = TransformGeometry(tree, location=(35, 2.6), translation=stand,
                                 rotation=[self.tape_tilt, 0, 0], name="LayTapeBack")
        create_geometry_line(tree, [joined, tilt])

        frame = Frame(tree, location=(25.6, 3.4), label="Cells")
        frame.add([dials, quad, fill, instances, realize, value, here, holds, under,
                   joined, tilt] + painters + drift)
        return tilt.geometry_out

    # ----------------------------------------------------------------
    def _create_cell_values(self, tree, control, variables, run):
        """``CellValues``: the number every cell holds, written on it.

        :param variables: unused here, and *run* only for its geometry; the
            transition machine writes the program and what the machine printed
            onto the cells and needs both - see
            :meth:`BrainFuckTransitionModifier._create_cell_values`.
        :return: the geometry socket of the numbers.
        """
        tape = run["Geometry"]
        dials = self._unpack(tree, control, "CellSize", "GlyphColor",
                             location=(25, -1.4), name="ValuesControl")
        value = NamedAttribute(tree, location=(26, -2), data_type="INT", name="Value")
        position = Position(tree, location=(26, -2.6))
        zone = ForEachZone(tree, location=(27, -1.4), domain="POINT", node_width=6,
                           geometry=tape)
        zone.add_socket(socket_type="INT", name="Value", value=value.std_out,
                        for_input=True)
        zone.add_socket(socket_type="VECTOR", name="Location", value=position.std_out,
                        for_input=True)

        digits = ValueToString(tree, location=(28, -0.8), data_type="INT",
                               value=zone.foreach_input.outputs["Value"], name="CellValue")
        size = MathNode(tree, location=(28, -2.4), operation="MULTIPLY",
                        inputs0=dials.out("CellSize"), inputs1=self.glyph_size,
                        name="NumberSize")
        curves = StringToCurves(tree, location=(29, -1.4), string=digits.std_out,
                                size=size.std_out, align_x="CENTER", align_y="BOTTOM")
        realize = RealizeInstances(tree, location=(30, -1.4))
        fill = FillCurve(tree, location=(31, -1.4), mode="N-gons")
        painted = SetMaterial(tree, location=(32, -1.4),
                              material=dials.out("GlyphColor"), name="PaintNumber")
        # the whole tape is laid back by tape_tilt further downstream, so a
        # number turned by the complement of that angle ends up standing
        # upright on a cell that is itself leaning away from the camera
        placed = TransformGeometry(tree, location=(33, -1.4),
                                   translation=zone.foreach_input.outputs["Location"],
                                   rotation=[pi / 2 - self.tape_tilt, 0, 0],
                                   name="PlaceNumber")
        zone.create_geometry_line([realize, fill, painted, placed],
                                  ins=curves.geometry_out)

        frame = Frame(tree, location=(25.6, -0.6), label="CellValues")
        frame.add([dials, value, position, zone, digits, size, curves, realize,
                   fill, painted, placed])
        return zone.geometry_out

    # ----------------------------------------------------------------
    def _create_code_table_frame(self, tree, control):
        """``CodeTable``: ``A``…1 to ``Z``…26, framed.

        A repeat zone walks the table one character at a time - again because
        ``Slice String`` needs a single index - and joins the letter and its
        number into the geometry it carries. The frame around the result is a
        rectangle sized from the bounding box of what came out, so it fits
        whatever alphabet is passed in.

        :return: the geometry socket of the table.
        """
        dials = self._unpack(tree, control, "CodeTable", "TablePosition",
                             "GlyphColor", location=(-15.4, 17.8),
                             name="TableControl")
        table = dials.out("CodeTable")
        size = StringLength(tree, location=(-14.4, 16.6), string=table,
                            name="TableLength")
        zone = RepeatZone(tree, location=(-13, 16), node_width=8,
                          iterations=size.std_out)

        # entry *n* stands ``table_spacing`` further right than entry *n-1*,
        # with the number on the line of TablePosition and the letter one
        # ``table_line_gap`` below it
        places = make_function(
            tree, name="EntryPosition", location=(-11, 17.4), hide=False,
            aux_functions={"across": "origin_x,column,%s,*,+" % self.table_spacing},
            functions={"NumberAt": ["across", "origin_y", "origin_z"],
                       "LetterAt": ["across", "origin_y",
                                    "origin_z,%s,-" % self.table_line_gap]},
            inputs=["origin", "column"], outputs=["NumberAt", "LetterAt"],
            vectors=["origin", "NumberAt", "LetterAt"], scalars=["across"],
            integers=["column"])
        tree.links.new(dials.out("TablePosition"), places.inputs["origin"])
        tree.links.new(zone.iteration, places.inputs["column"])

        letter = SliceString(tree, location=(-12, 14.6), string=table,
                             position=zone.iteration, length=1, name="Letter")
        letter_curves = StringToCurves(tree, location=(-11, 14.6), string=letter.std_out,
                                       size=self.table_glyph_size, align_x="CENTER",
                                       align_y="MIDDLE", hide=True)
        # the table is read as "A is 1", so the label is the 1-based index
        rank = IntegerMath(tree, location=(-12, 15.8), operation="ADD",
                           inputs0=zone.iteration, inputs1=1, name="Rank")
        number = ValueToString(tree, location=(-11, 15.8), data_type="INT",
                               value=rank.std_out, name="RankLabel")
        number_curves = StringToCurves(tree, location=(-10, 15.8), string=number.std_out,
                                       size=self.table_glyph_size, align_x="CENTER",
                                       align_y="MIDDLE", hide=True)

        entries, ends = [], []
        for curves, position, row, label in (
                (number_curves, places.outputs["NumberAt"], 17.4, "Number"),
                (letter_curves, places.outputs["LetterAt"], 14.6, "Letter")):
            # String to Curves hands out instances of outlines; realizing and
            # filling them turns them into the solid letter that is drawn
            realize = RealizeInstances(tree, location=(-8, row))
            fill = FillCurve(tree, location=(-7, row), mode="N-gons")
            # the entry is one piece of geometry, not a field, so it can be
            # moved with Transform Geometry - Set Position would need it to be
            # an instance first and would then have to be realized again
            place = TransformGeometry(tree, location=(-6, row),
                                      translation=position,
                                      rotation=[pi / 2, 0, 0], name="Place" + label)
            create_geometry_line(tree, [realize, fill, place], ins=curves.geometry_out)
            entries += [realize, fill, place]
            ends.append(place)

        pair = JoinGeometry(tree, location=(-5, 16))
        for end in ends:
            tree.links.new(end.geometry_out, pair.geometry_in)
        grown = JoinGeometry(tree, location=(-4.4, 16))
        tree.links.new(pair.geometry_out, grown.geometry_in)
        tree.links.new(zone.repeat_input.outputs["Geometry"], grown.geometry_in)
        tree.links.new(grown.geometry_out, zone.repeat_output.inputs["Geometry"])

        box = self._create_table_frame(tree, control, zone.geometry_out)
        joined = JoinGeometry(tree, location=(-1.6, 16))
        tree.links.new(zone.geometry_out, joined.geometry_in)
        tree.links.new(box, joined.geometry_in)
        painted = SetMaterial(tree, location=(-1, 16),
                              material=dials.out("GlyphColor"), name="PaintTable")
        create_geometry_line(tree, [joined, painted])

        frame = Frame(tree, location=(-14.6, 18.4), label="CodeTable")
        frame.add([dials, size, zone, places, letter, letter_curves, rank,
                   number, number_curves, pair, grown, joined, painted] + entries)
        return painted.geometry_out

    # ----------------------------------------------------------------
    def _create_table_frame(self, tree, control, table, location=(-3, 14.6),
                            label="TableFrame"):
        """The rectangle around the code table, sized from what it contains.

        :param location: where the nodes go in the editor. The transition
            machine draws a second table and needs a second box, somewhere
            else on the sheet.
        :return: the geometry socket of the rectangle.
        """
        x0, y0 = location

        def at(dx, dy):
            return x0 + dx, y0 + dy

        dials = self._unpack(tree, control, "FrameColor", location=at(7, -1.6),
                             name=label + "Control")
        bounds = BoundingBox(tree, location=at(0, 0), geometry=table)
        # the table stands in the x-z plane, so its width and height are the x
        # and z of the bounding box - grown by ``table_margin`` so the frame
        # stands off the lettering - and the middle of the box is where the
        # rectangle goes. ``low`` and ``high`` rather than min and max: those
        # two are operator tokens of the formula language.
        sides = make_function(
            tree, name=label + "Box", location=at(2, 0), hide=False,
            functions={"Width": "high_x,low_x,-,%s,*" % self.table_margin,
                       "Height": "high_z,low_z,-,%s,*" % self.table_margin,
                       "Centre": ["low_x,high_x,+,0.5,*", "low_y,high_y,+,0.5,*",
                                  "low_z,high_z,+,0.5,*"]},
            inputs=["low", "high"], outputs=["Width", "Height", "Centre"],
            vectors=["low", "high", "Centre"], scalars=["Width", "Height"])
        tree.links.new(bounds.min_out, sides.inputs["low"])
        tree.links.new(bounds.max_out, sides.inputs["high"])
        box = Quadrilateral(tree, location=at(4, 0), mode="RECTANGLE",
                            width=sides.outputs["Width"],
                            height=sides.outputs["Height"])
        # a bare curve renders as a hair thin enough to disappear, so the
        # rectangle is given a body before it is drawn
        wire = CurveWireFrame(tree, location=at(5, 0), radius=self.frame_radius,
                              resolution=4, geometry=box.geometry_out)
        place = TransformGeometry(tree, location=at(6, 0),
                                  translation=sides.outputs["Centre"],
                                  rotation=[pi / 2, 0, 0], name="Place" + label)
        painted = SetMaterial(tree, location=at(7, 0),
                              material=dials.out("FrameColor"),
                              name="Paint" + label)
        create_geometry_line(tree, [wire, place, painted])

        frame = Frame(tree, location=at(-1.4, 1.2), label=label)
        frame.add([dials, bounds, sides, box, wire, place, painted])
        return painted.geometry_out

    # ----------------------------------------------------------------
    def _create_display_frame(self, tree, control, label, width, position,
                              location, control_at):
        """One of the three framed boxes below the tape.

        :param width: the name of the control parameter holding the width of
            the box
        :param position: the name of the one holding the middle of it
        :param location: where the frame goes in the node editor
        :param control_at: where its Separate Bundle goes. The program strip is
            drawn across the same patch of the editor as the two displays and
            leaves no one offset from *location* that is clear for both.
        :return: the geometry socket of the box.
        """
        x, y = location
        dials = self._unpack(tree, control, width, position, "FrameColor",
                             location=control_at, name=label + "Control")
        box = Quadrilateral(tree, location=(x - 1, y), mode="RECTANGLE",
                            width=dials.out(width), height=self.display_height)
        # a bare curve renders as a hair thin enough to disappear, so the
        # rectangle is given a body before it is drawn
        increase_resolution = ResampleCurve(tree, location=(x, y), count=1000, curve=box.geometry_out)
        wire = CurveWireFrame(tree, location=(x + 1, y), radius=self.frame_radius,
                              resolution=4, geometry=increase_resolution.geometry_out)
        place = TransformGeometry(tree, location=(x + 2, y),
                                  translation=dials.out(position),
                                  rotation=[pi / 2, 0, 0], name="Place" + label)
        painted = SetMaterial(tree, location=(x + 3, y),
                              material=dials.out("FrameColor"),
                              name="Paint" + label)
        create_geometry_line(tree, [place, painted], ins=wire.geometry_out)

        frame = Frame(tree, location=(x - 0.4, y + 0.8), label=label)
        frame.add([dials, box, wire, place, painted])
        return painted.geometry_out

    # ----------------------------------------------------------------
    def _create_program_strip(self, tree, control, variables, run):
        """``ProgramStrip``: the whole program, written out once and left alone.

        The program is drawn one instruction per column, spread across the
        input display, and it does not move. What moves is ``CurrentDisplay``,
        a box that runs along the strip and stands around the instruction the
        counter points at. So the program is read where a program is normally
        read - in one piece, in one place - and the machine is what travels
        over it. A loop then shows as the box running back to the ``[`` and
        crossing the same instructions again.

        Columns are of one width whatever stands in them, so the strip is a
        ruler and the box's position is the program counter to scale.

        Each instruction is painted first by *what it is* - its colour from
        :data:`INSTRUCTION_COLORS`, the palette every scene of the video shares,
        so that the ``<`` of this strip is the ``<`` of the soup watcher's tape
        - and then by *what has become of it*, see :attr:`PROGRAM_COLORS`. The
        second one worth the trouble is ``WaitingColor``: an instruction that
        has run but sits inside a loop that is still open will run again, and
        :meth:`_loop_starts` says which those are. What is left in
        ``DoneColor`` is what has run for the last time, so the strip goes dark
        behind the box only where the machine is never coming back - and what
        is still in its own colour ahead of the box is what has not run yet.

        The box marks the instruction *about to* run, not the one just run -
        the tape beside it is the state that instruction is about to act on,
        which is how a debugger shows the same thing.

        :return: the geometry socket of the strip and of the box on it.
        """
        source = self._unpack(tree, variables, "Input", "Loops",
                              location=(16, -21.4), name="StripVariables")
        program = source.out("Input")
        counter = run["Counter"]
        # the strip is the hungriest frame of the graph: where the display it
        # is written across stands, and thirteen colours - one per instruction,
        # two for what has become of one, and the colour of the head
        dials = self._unpack(tree, control, "InputPosition", "InputDisplaySize",
                             *[node_name for node_name, _, _ in self.opcode_colors],
                             *[node_name for node_name, _ in self.program_colors],
                             "CurrentColor", location=(17, -25.0),
                             name="StripControl")
        size = StringLength(tree, location=(17, -21.4), string=program,
                            name="StripLength")

        # --- where the strip sits ---------------------------------------
        # A column per instruction, plus one at each end. The first is the
        # margin that keeps column 0 clear of the left edge; the second is
        # where the counter ends up when the program has run out, and the box
        # that marks it needs somewhere to park that is still inside the
        # display rather than astride its right edge. ``First`` is the middle
        # of column 0, one such gap in from the left edge of the display.
        ruler = make_function(
            tree, name="StripLayout", location=(19.4, -22.2), hide=False,
            aux_functions={"gap": "width,%d,/" % (len(self.program) + 2)},
            functions={"First": "place_x,width,0.5,*,-,gap,+", "Spacing": "gap",
                       "Glyph": "gap,%s,*" % self.strip_glyph_size},
            inputs=["place", "width"], outputs=["First", "Spacing", "Glyph"],
            vectors=["place"],
            scalars=["width", "gap", "First", "Spacing", "Glyph"])
        tree.links.new(dials.out("InputPosition"), ruler.inputs["place"])
        tree.links.new(dials.out("InputDisplaySize"), ruler.inputs["width"])
        first, spacing = ruler.outputs["First"], ruler.outputs["Spacing"]

        # --- which loop is open where the counter stands -----------------
        entry = SliceString(tree, location=(17, -23.8), string=source.out("Loops"),
                            position=counter, length=1, name="LoopEntry")
        encoded = CharToAscii(tree, location=(18, -23.8), char=entry.std_out,
                              name="LoopCode")
        # --- one column per instruction ----------------------------------
        zone = RepeatZone(tree, location=(21, -21.4), node_width=9,
                          iterations=size.std_out)
        column = zone.iteration
        letter = SliceString(tree, location=(22, -22.2), string=program,
                             position=column, length=1, name="StripLetter")
        curves = StringToCurves(tree, location=(23, -22.2), string=letter.std_out,
                                size=ruler.outputs["Glyph"], align_x="CENTER",
                                align_y="MIDDLE", hide=True)
        realize = RealizeInstances(tree, location=(24, -22.2))
        fill = FillCurve(tree, location=(25, -22.2), mode="N-gons")
        # column n stands n spacings in from the first one, and stays there
        at = make_function(
            tree, name="ColumnPlace", location=(23.6, -20.8), hide=False,
            functions={"At": ["first,column,spacing,*,+", "place_y", "place_z"]},
            inputs=["first", "spacing", "column", "place"], outputs=["At"],
            scalars=["first", "spacing"], integers=["column"],
            vectors=["place", "At"])
        for socket, socket_name in ((first, "first"), (spacing, "spacing"),
                                    (column, "column"),
                                    (dials.out("InputPosition"), "place")):
            tree.links.new(socket, at.inputs[socket_name])
        place = TransformGeometry(tree, location=(26, -22.2),
                                  translation=at.outputs["At"],
                                  rotation=[pi / 2, 0, 0], name="PlaceColumn")

        # What has become of this column, from the loop table read above:
        #
        # ``opened``
        #     the outermost ``[`` still open where the counter stands, 1-based,
        #     so 0 means it is not inside a loop at all and nothing is waiting.
        # ``Waits``
        #     this column has run and is inside that loop, so it will run
        #     again. The "[" itself is not re-executed - "]" jumps back to the
        #     instruction after it - so the block that is waiting starts one
        #     column further on, which is what ``column >= opened`` says.
        state = make_function(
            tree, name="ColumnState", location=(23, -25.4), hide=False,
            aux_functions={"opened": "code,%d,-" % self.JUMP_ORIGIN,
                           "done": "column,counter,<"},
            functions={"Done": "done", "Now": "column,counter,=",
                       "Waits": "opened,0,>,column,opened,<,not,and,done,and"},
            inputs=["code", "column", "counter"],
            outputs=["Done", "Waits", "Now"],
            integers=["code", "column", "counter"], scalars=["opened", "done"],
            booleans=["Done", "Waits", "Now"])
        for socket, socket_name in ((encoded.std_out, "code"), (column, "column"),
                                    (counter, "counter")):
            tree.links.new(socket, state.inputs[socket_name])
        done, waits, now = (state.outputs["Done"], state.outputs["Waits"],
                            state.outputs["Now"])

        # what the instruction is, before what has become of it: one Set
        # Material per entry of the shared palette, each selecting on "this
        # column holds one of my characters"
        which = instruction_selector(tree, letter.std_out, self.opcode_colors,
                                     location=(26, -24.6), name="ColorSelector")
        painters = [SetMaterial(tree, location=(27, -22.2 - 0.3 * row),
                                selection=which.outputs[node_name],
                                material=dials.out(node_name),
                                name="Paint" + node_name, hide=True)
                    for row, (node_name, _, _) in enumerate(self.opcode_colors)]

        # ... and then what has become of it, which overrides it
        selections = (done, waits)
        painters += [SetMaterial(tree, location=(28 + step, -22.2), selection=selection,
                                 material=dials.out(node_name),
                                 name="Paint" + node_name)
                     for step, ((node_name, _), selection)
                     in enumerate(zip(self.program_colors, selections))]
        painters.append(SetMaterial(tree, location=(30, -22.2), selection=now,
                                    material=dials.out("CurrentColor"),
                                    name="PaintCurrentInstruction"))
        create_geometry_line(tree, [realize, fill, place] + painters,
                             ins=curves.geometry_out)

        grown = JoinGeometry(tree, location=(31, -22.2))
        tree.links.new(painters[-1].geometry_out, grown.geometry_in)
        tree.links.new(zone.repeat_input.outputs["Geometry"], grown.geometry_in)
        tree.links.new(grown.geometry_out, zone.repeat_output.inputs["Geometry"])

        frame = Frame(tree, location=(16.6, -20.6), label="ProgramStrip")
        frame.add([dials, source, size, ruler, entry, encoded, zone, letter,
                   which, curves, realize, fill, at, place, state,
                   grown] + painters)

        cursor = self._create_cursor_frame(tree, control, counter, first, spacing,
                                           dials.out("InputPosition"))
        both = JoinGeometry(tree, location=(33, -22.2))
        for piece in (zone.geometry_out, cursor):
            tree.links.new(piece, both.geometry_in)
        frame.add([both])
        return both.geometry_out

    # ----------------------------------------------------------------
    def _create_cursor_frame(self, tree, control, counter, first, spacing, place):
        """``CurrentDisplay``: the box that runs along the program strip.

        The same framed rectangle the read-outs are drawn with, sized to a
        single column of :meth:`_create_program_strip` and put where the
        counter points instead of somewhere fixed. It is painted in
        ``PointerColor``, the colour of the marker under the tape, so that the
        two heads - the one on the program and the one on the data - read as
        the same thing.

        :param counter: the program counter
        :param first: x of column 0 of the strip
        :param spacing: width of one column
        :param place: the middle of the display the strip is written across
        :return: the geometry socket of the box.
        """
        dials = self._unpack(tree, control, "CurrentColor", location=(21, -30.0),
                             name="CursorControl")
        # the box stands where the counter points, which is column 0 plus so
        # many columns along, and is a little wider than a column so that it
        # stands around the instruction rather than on it
        at = make_function(
            tree, name="CursorPlace", location=(22, -27.8), hide=False,
            functions={"At": ["first,counter,spacing,*,+", "place_y", "place_z"],
                       "Wide": "spacing,%s,*" % self.cursor_width},
            inputs=["first", "spacing", "counter", "place"],
            outputs=["At", "Wide"],
            scalars=["first", "spacing", "Wide"], integers=["counter"],
            vectors=["place", "At"])
        for socket, socket_name in ((first, "first"), (spacing, "spacing"),
                                    (counter, "counter"), (place, "place")):
            tree.links.new(socket, at.inputs[socket_name])
        box = Quadrilateral(tree, location=(24, -28.2), mode="RECTANGLE",
                            width=at.outputs["Wide"],
                            height=self.cursor_height * self.display_height)
        # a bare curve renders as a hair thin enough to disappear
        wire = CurveWireFrame(tree, location=(25, -28.2), radius=self.frame_radius,
                              resolution=4, geometry=box.geometry_out)
        put = TransformGeometry(tree, location=(26, -28.2),
                                translation=at.outputs["At"],
                                rotation=[pi / 2, 0, 0], name="PlaceCursor")
        painted = SetMaterial(tree, location=(27, -28.2),
                              material=dials.out("CurrentColor"),
                              name="PaintCursor")
        create_geometry_line(tree, [put, painted], ins=wire.geometry_out)

        frame = Frame(tree, location=(20.6, -26.6), label="CurrentDisplay")
        frame.add([dials, at, box, wire, put, painted])
        return painted.geometry_out

    # ----------------------------------------------------------------
    def _create_simulated_geometry_frame(self, tree, control, variables, run):
        """``SimulatedGeometry``: everything that is redrawn every frame.

        What the machine has printed, written into its box, and the marker
        under the cell the head is on. This is built from the state the
        simulation zone *outputs*, not inside the zone: none of it is state, it
        is a picture of the state.

        :return: the geometry socket.
        """
        label, y = "OutputText", -8
        dials = self._unpack(tree, control, "OutputDisplaySize", "OutputPosition",
                             "PointerOffset", "TapePosition", "PointerColor",
                             "GlyphColor", location=(26, -11.6),
                             name="SimulatedControl")
        box_width, position = dials.out("OutputDisplaySize"), dials.out("OutputPosition")
        # Plain text centred on the origin, *not* String to Curves' own
        # SCALE_TO_FIT: a text box hangs off the origin rather than surrounding
        # it, and where inside the box the text ends up moves with how far it
        # had to be shrunk - a long string comes out below the box that a two
        # letter one sits in the middle of. Centred text is in the same place
        # whatever it says, and the fitting is done below, where it can be
        # measured.
        printed_text = self._output_string(tree, control, run["Output"])
        curves = StringToCurves(tree, location=(26, y), string=printed_text,
                                size=0.6 * self.display_height, align_x="CENTER",
                                align_y="MIDDLE", name=label, hide=True)
        realize = RealizeInstances(tree, location=(27, y))
        fill = FillCurve(tree, location=(28, y), mode="N-gons")
        # how much wider than its box the text came out, as the scale that
        # brings it back inside:
        #
        # ``wide``
        #     the width of what was drawn. An empty string has no geometry and
        #     hence no width, and the guard keeps the division finite - the
        #     ``min`` below then leaves it alone at scale 1.
        # ``factor``
        #     only ever shrink: a short output should not be blown up to the
        #     full width of its box.
        bounds = BoundingBox(tree, location=(29, y - 1.4))
        fit = make_function(
            tree, name="Fit" + label, location=(32, y - 1.4), hide=False,
            aux_functions={"wide": "high_x,low_x,-,0.001,max",
                           "factor": "box,wide,/,1,min"},
            functions={"Scale": ["factor", "factor", "factor"]},
            inputs=["low", "high", "box"], outputs=["Scale"],
            vectors=["low", "high", "Scale"],
            scalars=["box", "wide", "factor"])
        for socket, socket_name in ((bounds.min_out, "low"), (bounds.max_out, "high"),
                                    (box_width, "box")):
            tree.links.new(socket, fit.inputs[socket_name])
        place = TransformGeometry(tree, location=(29, y), translation=position,
                                  rotation=[pi / 2, 0, 0],
                                  scale=fit.outputs["Scale"], name="Place" + label)
        create_geometry_line(tree, [realize, fill, place], ins=curves.geometry_out)
        tree.links.new(fill.geometry_out, bounds.geometry_in)
        pieces = [curves, realize, fill, bounds, fit, place]
        written = [place]

        # --- the head marker -------------------------------------------
        # the x of the cell is read off the tape rather than recomputed from
        # TapeSize and CellSize, so the marker cannot drift away from the cells
        # if the spacing of the Mesh Line is ever changed
        at = Position(tree, location=(26, -14))
        spot = SampleIndex(tree, location=(27, -14), data_type="FLOAT_VECTOR",
                           domain="POINT", geometry=run["Geometry"], value=at.std_out,
                           index=run["PointerPosition"], name="CellPosition")
        # under the cell the head is on: its x, and the y and z of the offset
        # that hangs the marker below the tape - plus ``TapePosition``, which
        # is what moves the cells (it rides on ``LayTapeBack``), and the
        # marker stands under one of them
        under = make_function(
            tree, name="MarkerPosition", location=(28.4, -14), hide=False,
            functions={"Under": ["spot_x,shift_x,+", "drop_y,shift_y,+",
                                 "drop_z,shift_z,+"]},
            inputs=["spot", "drop", "shift"], outputs=["Under"],
            vectors=["spot", "drop", "shift", "Under"])
        stand, drift = self._tape_stand(tree, control, dials.out("TapePosition"),
                                        location=(26.8, -12.4), name="MarkerStand")
        tree.links.new(spot.std_out, under.inputs["spot"])
        tree.links.new(dials.out("PointerOffset"), under.inputs["drop"])
        tree.links.new(stand, under.inputs["shift"])
        # an arrow pointing up at the cell, short enough to stay in the gap
        # between the tape and the read-outs below it
        tip = ConeMesh(tree, location=(26, -16), vertices=32, radius_top=0,
                       radius_bottom=0.2 * self.cell_size, depth=0.5 * self.cell_size)
        stem = CylinderMesh(tree, location=(26, -17), vertices=32,
                            radius=0.1 * self.cell_size, depth=0.5 * self.cell_size)
        lowered = TransformGeometry(tree, location=(27, -17),
                                    translation=[0, 0, 0],
                                    name="StemBelowTip")
        create_geometry_line(tree, [stem, lowered])
        marker = JoinGeometry(tree, location=(28, -16))
        tree.links.new(tip.geometry_out, marker.geometry_in)
        tree.links.new(lowered.geometry_out, marker.geometry_in)
        put = TransformGeometry(tree, location=(29, -16),
                                translation=under.outputs["Under"],
                                name="PlaceMarker")
        painted = SetMaterial(tree, location=(30, -16),
                              material=dials.out("PointerColor"),
                              name="PaintMarker")
        create_geometry_line(tree, [marker, put, painted])

        # the three strings are painted together, and only then joined with the
        # marker: a Set Material without a selection paints everything it is
        # handed, so putting the marker in first would take its colour away
        lettering = JoinGeometry(tree, location=(33, -10))
        for piece in written:
            tree.links.new(piece.geometry_out, lettering.geometry_in)
        text = SetMaterial(tree, location=(34, -10),
                           material=dials.out("GlyphColor"), name="PaintText")
        create_geometry_line(tree, [lettering, text])

        # the strip carries its own colours, one per instruction, so it joins
        # after the painting rather than before it
        strip = self._create_program_strip(tree, control, variables, run)
        joined = JoinGeometry(tree, location=(35, -12))
        for piece in (text.geometry_out, painted.geometry_out, strip):
            tree.links.new(piece, joined.geometry_in)

        frame = Frame(tree, location=(25.6, -7.2), label="SimulatedGeometry")
        frame.add(pieces + [dials, at, spot, under, tip, stem, lowered,
                            marker, put, painted, lettering, text,
                            joined] + drift)
        return joined.geometry_out


class BrainFuckTransitionModifier(BrainFuckSimpleModifier):
    """The same machine, with a tape that can change size while it runs.

    :class:`BrainFuckSimpleModifier` seeds its tape into the simulation zone
    once and hands it from frame to frame, which is what keeps the cell values
    alive from one instruction to the next. The price is that the *shape* of
    the tape is settled on the frame the simulation starts: ``TapeSize`` and
    ``CellSize`` are read by a ``Mesh Line`` standing in front of the zone,
    and the zone never looks at it again. Animating them after that does
    nothing - or worse than nothing, since ``CellSize`` also sizes the squares
    of the ``Cells`` frame, which are drawn outside the zone and do follow: the
    cells shrink while their spacing stays where it was.

    This machine rebuilds the tape inside the zone instead, once per frame -
    see :meth:`_tape_in_zone`. A fresh ``Mesh Line`` at the current
    ``TapeSize`` and ``CellSize``, with the value of every cell carried over
    from the tape of the frame before, by index; a cell that was not there
    before starts at zero. So the tape can grow, shrink and re-space while the
    machine runs and what is written on it survives, which is what the
    ``bf_to_bff`` transition needs: five fat cells opening out into sixty-four
    thin ones without the machine stopping to do it.

    Everything else is the simple machine, deliberately. The code table, the
    two read-outs and the program strip are laid out in python from the
    ``tape_size`` and ``cell_size`` the modifier was *built* with, and they
    stay where they were built - so a tape that has grown past them is moved
    back inside the picture with ``TapePosition``, which rides on
    ``LayTapeBack`` at the end of the ``Cells`` frame and on the head marker
    with it. That is out beyond the zone, where an animated value still has an
    effect.

    Two things to know before animating it:

    - The values are carried over **by index**, so cell *n* keeps what it
      holds only as long as it is still cell *n*. A tape that shrinks past the
      head drops what was beyond the new end, which is what shrinking a tape
      means. Growing never loses anything.
    - ``TapeSize`` is read on the frame the simulation starts as well as on
      every frame after it, so a growth that begins at the same moment the
      machine does starts from wherever the interpolation has reached on the
      first frame, not from the value at frame zero.
    **The copy of the program, and the letters going onto the tape.** The
    shot ends by turning the program into data, which is the whole of the
    difference between brainfuck and BFF: there the tape *is* the program.
    Three things happen in order, and each has its own parameter so the scene
    can place it in time:

    ``CopyProgramTime``
        a second copy of the program strip appears - the same instructions in
        the same colours, drawn by ``CopyOfProgramStrip``. Until then it is
        deleted rather than hidden, so it costs nothing.
    ``ProgramShift``, ``ShrinkFontSize``, ``ShrinkSpacing``
        the copy is moved and squeezed until it stands over the tape at the
        pitch of the cells. The two shrink factors ride inside
        ``StripLayout2``, the copy's own layout formula.
    ``SwitchLetterTime``, ``LetterDuration``
        from then on one letter per ``LetterDuration`` leaves the copy and
        appears on the cell below it, left to right - see
        :meth:`_letters_landed`. A cell that has been written to shows the
        instruction instead of its number; a column that has been written
        away is gone from the copy. When the last one has landed the tape
        reads as the program, which is where the next scene starts.

    The cell *values* are not touched by this - what changes is what is drawn
    on them. Writing the ascii codes into the tape as well would be a
    ``Store Named Attribute`` inside the zone, and would have to wait until
    the machine has stopped.

    :param copy_program_time: seconds before the copy of the program appears
    :param switch_letter_time: seconds before its first letter lands on the
        tape
    :param letter_duration: seconds between one letter landing and the next
    """

    #: how far past the edges of the input display the wipe of
    #: :meth:`_program_wipe` starts and finishes, so that the frame around the
    #: display goes with what is inside it rather than a moment later
    WIPE_MARGIN = 0.2

    #: the arrow the output display turns into, as
    #: :class:`~geometry_nodes.nodes.MorphNode2` wants it: a curve and a
    #: radius. ``SAMPLES`` is what both shapes are resampled to, so it also
    #: sets how sharp the barbs come out and how wide the notch is where the
    #: frame is cut open to be a curve with two ends.
    ARROW_SAMPLES = 192
    ARROW_PROFILE = 16
    ARROW_SHAFT_RADIUS = 0.17
    ARROW_HEAD_RADIUS = 0.34
    ARROW_SHAFT_LENGTH = 0.5
    ARROW_HEAD_LENGTH = 0.5
    #: how far the point of the arrow floats above the cell it marks
    ARROW_GAP = 0.5
    #: what the display is painted once it is on its way to being an arrow
    ARROW_COLOR = "custom1"

    #: the last transform: the alphabet of the simple machine leaves to the
    #: right and the ascii table of the extended one comes in from the left,
    #: while the tape and the two arrows that point at it step down out of the
    #: way. Far enough that both tables are off the edge of the shot, which is
    #: about 26 units wide at the camera the scene sets up.
    TABLE_SLIDE = 30.0
    TAPE_DROP = -3

    #: the ascii table the extended machine reads, as this shot draws it: the
    #: printable characters over their codes, wrapped into bands of
    #: ``ASCII_WIDTH``. The ten instructions are drawn ``ASCII_COMMAND_SCALE``
    #: times the size of the rest and in their own colours, which is the whole
    #: point of showing it - the alphabet that is being left behind had
    #: twenty-six letters and no instructions in it.
    ASCII_FIRST, ASCII_LAST = 32, 126
    ASCII_WIDTH = 32
    ASCII_SPACING = 0.58
    ASCII_GLYPH = 0.26
    ASCII_COMMAND_SCALE = 1.9
    ASCII_LINE_GAP = 0.5
    ASCII_BAND_GAP = 0.45
    #: where the top line of codes stands. The grid hangs down from here, and
    #: three bands of it have to finish above the arrow on the dropped tape.
    ASCII_TOP = 3.6

    def __init__(self, copy_program_time=5.0, switch_letter_time=7.0,
                 letter_duration=0.1, program_disappear_time=10.0,
                 program_disappear_duration=1.0, output_offset=32,
                 output_disappear_time=12.0, output_move_duration=1.5,
                 output_recolor_time=13.0,
                 output_morph_time=14.0, output_morph_duration=2.0,
                 replace_code_table=18.0, replace_code_table_duration=1.0,
                 **kwargs):
        self.copy_program_time = copy_program_time
        self.switch_letter_time = switch_letter_time
        self.letter_duration = letter_duration
        self.program_disappear_time = program_disappear_time
        self.program_disappear_duration = program_disappear_duration
        self.output_offset = output_offset
        self.output_disappear_time = output_disappear_time
        self.output_move_duration = output_move_duration
        self.output_recolor_time = output_recolor_time
        self.output_morph_time = output_morph_time
        self.output_morph_duration = output_morph_duration
        self.replace_code_table = replace_code_table
        self.replace_code_table_duration = replace_code_table_duration
        # both of these are asked for more than once and built once
        self._landed = None
        self._wipe_edge = None
        self._wipe_frame = None
        self._swap = None
        super().__init__(**kwargs)

    # ----------------------------------------------------------------
    @property
    def output(self):
        """What the machine prints, worked out in python.

        A property rather than something ``__init__`` computes, because the
        graph is built by ``super().__init__`` and wants this while it is
        being built - by which time the program is known but the constructor
        has not come back yet.
        """
        return self.simulate(self.program, self.tape_size, self.code_table)[1]

    @property
    def arrow_cell(self):
        """The cell the arrow points at: the last letter the machine printed.

        ``HELLO`` written from :attr:`output_offset` puts its ``O`` on cell
        ``output_offset + 4``, and that is where the arrow comes to rest.
        """
        return self.output_offset + max(len(self.output), 1) - 1

    @property
    def barb_fraction(self):
        """Where along the arrow's axis the shaft stops and the head begins."""
        return self.ARROW_SHAFT_LENGTH / (self.ARROW_SHAFT_LENGTH
                                          + self.ARROW_HEAD_LENGTH)

    # ----------------------------------------------------------------
    def _more_control(self, tree, control, x):
        """The six parameters of the transition, added to the control frame.

        ``CopyProgramTime``, ``SwitchLetterTime`` and ``LetterDuration`` are
        set when the modifier is built and left alone; ``ProgramShift`` and
        the two shrink factors are what the scene keyframes to bring the copy
        of the program down onto the tape.
        """
        for row, (node_name, value) in enumerate((
                ("CopyProgramTime", self.copy_program_time),
                ("ShrinkFontSize", 1.0),
                ("ShrinkSpacing", 1.0),
                ("SwitchLetterTime", self.switch_letter_time),
                ("LetterDuration", self.letter_duration),
                ("ProgramDisappearTime", self.program_disappear_time),
                ("ProgramDisappearDuration", self.program_disappear_duration),
                ("OutputDisappearTime", self.output_disappear_time),
                ("OutputMoveDuration", self.output_move_duration),
                ("OutputRecolorTime", self.output_recolor_time),
                ("OutputMorphTime", self.output_morph_time),
                ("OutputMorphDuration", self.output_morph_duration),
                ("ReplaceCodeTable", self.replace_code_table),
                ("ReplaceCodeTableDuration", self.replace_code_table_duration))):
            control[node_name] = InputValue(tree, location=(x, -12.6 - 0.4 * row),
                                            value=value, name=node_name)
        control["OutputOffset"] = InputInteger(tree, location=(x, -17.2),
                                               integer=self.output_offset,
                                               name="OutputOffset")
        control["ArrowColor"] = InputMaterial(tree, location=(x, -17.8),
                                              material=self.ARROW_COLOR,
                                              name="ArrowColor", **self.kwargs)
        self.materials.append(control["ArrowColor"].node.material)
        control["ProgramShift"] = InputVector(tree, location=(x, -14.8),
                                              vector=Vector(), name="ProgramShift")
        # The ascii table is a grid of 32 columns rather than a row of 26, so
        # it is centred on the tape by its own width, and it stands high
        # enough that its bottom band clears the arrow standing on the tape -
        # which is itself a tape-drop lower by then.
        middle = 0.5 * self.tape_size * self.cell_size
        control["AsciiTablePosition"] = InputVector(
            tree, location=(x, -18.6), name="AsciiTablePosition",
            vector=Vector([middle - 0.5 * (self.ASCII_WIDTH - 1) * self.ASCII_SPACING,
                           0, self.ASCII_TOP]))

    # ----------------------------------------------------------------
    def _table_swap(self, tree, control):
        """``TableSwap``: how far the last transform has got, from 0 to 1.

        The one number behind the whole of it - the alphabet leaving to the
        right, the ascii table arriving from the left, and the tape and its
        two arrows stepping down between them. Everything that moves reads
        this, so nothing can drift out of step with the rest.

        :return: a FLOAT socket, 0 before ``ReplaceCodeTable`` and 1
            ``ReplaceCodeTableDuration`` later.
        """
        if self._swap is not None:
            return self._swap
        dials = self._unpack(tree, control, "ReplaceCodeTable",
                             "ReplaceCodeTableDuration", location=(-9, 12.4),
                             name="SwapControl")
        now = SceneTime(tree, location=(-9, 11.4), std_out="Seconds",
                        name="SwapClock")
        gone = make_function(
            tree, name="TableSwapDone", location=(-7.6, 11.8), hide=False,
            functions={"Gone": "seconds,start,-,duration,/,0,max,1,min"},
            inputs=["seconds", "start", "duration"], outputs=["Gone"],
            scalars=["seconds", "start", "duration", "Gone"])
        for socket, socket_name in (
                (now.std_out, "seconds"),
                (dials.out("ReplaceCodeTable"), "start"),
                (dials.out("ReplaceCodeTableDuration"), "duration")):
            tree.links.new(socket, gone.inputs[socket_name])

        frame = Frame(tree, location=(-9.4, 13.2), label="TableSwap")
        frame.add([dials, now, gone])
        self._swap = gone.outputs["Gone"]
        return self._swap

    # ----------------------------------------------------------------
    def _tape_stand(self, tree, control, position, location=(0, 0),
                    name="TapeStand"):
        """``TapePosition``, stepping down as the tables change over.

        The ascii table of the extended machine is a grid rather than a row
        and wants the room, so the tape and the two arrows that point at it -
        the head marker below and the one the read-out became above - move
        down together over the same second. They all read this, so they move
        as one thing rather than three.
        """
        swap = self._table_swap(tree, control)
        drop = make_function(
            tree, name=name, location=location, hide=True,
            functions={"Down": ["place_x", "place_y",
                                "place_z,gone,%s,*,+" % self.TAPE_DROP]},
            inputs=["place", "gone"], outputs=["Down"],
            vectors=["place", "Down"], scalars=["gone"])
        tree.links.new(position, drop.inputs["place"])
        tree.links.new(swap, drop.inputs["gone"])
        return drop.outputs["Down"], [drop]

    # ----------------------------------------------------------------
    def _create_code_table_frame(self, tree, control):
        """The alphabet of the simple machine, on its way out to the right."""
        table = super()._create_code_table_frame(tree, control)
        swap = self._table_swap(tree, control)
        away = make_function(
            tree, name="TableLeaving", location=(-1, 11.6), hide=True,
            functions={"Away": ["gone,%s,*" % self.TABLE_SLIDE, "0", "0"]},
            inputs=["gone"], outputs=["Away"], scalars=["gone"],
            vectors=["Away"])
        tree.links.new(swap, away.inputs["gone"])
        slid = TransformGeometry(tree, location=(0.4, 11.0), geometry=table,
                                 translation=away.outputs["Away"],
                                 name="SlideOldTable")
        frame = Frame(tree, location=(-1.4, 13.0), label="TableLeaves")
        frame.add([away, slid])
        return slid.geometry_out

    # ----------------------------------------------------------------
    @property
    def ascii_table(self):
        """The printable characters, the ones the extended machine reads."""
        return "".join(chr(code) for code
                       in range(self.ASCII_FIRST, self.ASCII_LAST + 1))

    # ----------------------------------------------------------------
    def _create_ascii_table_frame(self, tree, control):
        """``AsciiTable``: what the machine after this one reads, arriving.

        The alphabet of the simple machine says ``A`` is 1 and stops at 26.
        The extended machine has no alphabet: its cells hold bytes, and the
        bytes that happen to be instructions are the program. So the table
        that replaces it is the printable half of ascii over the codes it
        stands for, wrapped into bands of :attr:`ASCII_WIDTH`, with the ten
        instructions drawn large and in their own colours so that the eye can
        find them in the crowd. That contrast is the argument the shot is
        making, which is why the two tables cross rather than cut.

        It is built here rather than taken from
        :class:`BrainFuckExtendedModifier`: that machine's table reads its own
        control frame - a ``TableWidth``, a ``CommandTable``, an ``OpColors``
        bundle - and calls a ``_create_table_frame`` of a different signature,
        so sharing it would mean restructuring a class this shot does not use.

        :return: the geometry socket of the table, on its way in from the left.
        """
        y = -53.0
        swap = self._table_swap(tree, control)
        dials = self._unpack(tree, control, "AsciiTablePosition", "GlyphColor",
                             *[node_name for node_name, _, _ in self.opcode_colors],
                             location=(17, y), name="AsciiControl")
        table = InputString(tree, location=(17, y - 1.4), string=self.ascii_table,
                            name="AsciiCharacters")
        commands = InputString(tree, location=(17, y - 2.0),
                               string="".join(characters for _, _, characters
                                              in self.opcode_colors),
                               name="AsciiCommands")

        zone = RepeatZone(tree, location=(19, y), node_width=11,
                          iterations=len(self.ascii_table))
        entry = zone.iteration
        # where this entry stands: along its band, and one band lower for
        # every ASCII_WIDTH characters. The code goes on the upper line and
        # the character it stands for on the lower one.
        places = make_function(
            tree, name="AsciiPlaces", location=(20.4, y - 0.6), hide=False,
            aux_functions={
                "column": "i,%d,%%" % self.ASCII_WIDTH,
                "band": "i,%d,/,floor" % self.ASCII_WIDTH,
                "across": "origin_x,column,%s,*,+" % self.ASCII_SPACING,
                "line": "origin_z,band,%s,*,-" % (2 * self.ASCII_LINE_GAP
                                                  + self.ASCII_BAND_GAP)},
            functions={"CodeAt": ["across", "origin_y", "line"],
                       "CharAt": ["across", "origin_y",
                                  "line,%s,-" % self.ASCII_LINE_GAP]},
            inputs=["origin", "i"], outputs=["CodeAt", "CharAt"],
            vectors=["origin", "CodeAt", "CharAt"], integers=["i"],
            scalars=["column", "band", "across", "line"])
        tree.links.new(dials.out("AsciiTablePosition"), places.inputs["origin"])
        tree.links.new(entry, places.inputs["i"])

        character = SliceString(tree, location=(20.4, y - 2.0), string=table.std_out,
                                position=entry, length=1, name="AsciiCharacter")
        # an instruction is drawn large: "in" counts the character inside the
        # set of instructions, which is 1 for one of the ten and 0 for the
        # rest, and the size follows from that
        big = make_function(
            tree, name="AsciiGlyphSize", location=(21.4, y - 2.0), hide=True,
            custom_ops={"in": {"type": FindInString,
                               "inputs": ("String", "Search"),
                               "output": "Count", "label": "in"}},
            functions={"size": "%s,1,commands,letter,in,0,>,%s,1,-,*,+,*"
                               % (self.ASCII_GLYPH, self.ASCII_COMMAND_SCALE)},
            inputs=["commands", "letter"], outputs=["size"],
            strings=["commands", "letter"], scalars=["size"])
        tree.links.new(commands.std_out, big.inputs["commands"])
        tree.links.new(character.std_out, big.inputs["letter"])
        char_curves = StringToCurves(tree, location=(22.4, y - 2.0),
                                     string=character.std_out,
                                     size=big.outputs["size"], align_x="CENTER",
                                     align_y="MIDDLE", hide=True)
        code = IntegerMath(tree, location=(20.4, y - 1.2), operation="ADD",
                           inputs0=entry, inputs1=self.ASCII_FIRST, name="AsciiCode")
        label = ValueToString(tree, location=(21.4, y - 1.2), data_type="INT",
                              value=code.std_out, name="AsciiCodeLabel")
        code_curves = StringToCurves(tree, location=(22.4, y - 1.2),
                                     string=label.std_out, size=self.ASCII_GLYPH,
                                     align_x="CENTER", align_y="MIDDLE", hide=True)

        entries, ends = [], []
        for curves, position, row, tag in (
                (code_curves, places.outputs["CodeAt"], y - 0.2, "AsciiCode"),
                (char_curves, places.outputs["CharAt"], y - 3.0, "AsciiChar")):
            realize = RealizeInstances(tree, location=(23.4, row))
            fill = FillCurve(tree, location=(24.4, row), mode="N-gons")
            place = TransformGeometry(tree, location=(25.4, row),
                                      translation=position,
                                      rotation=[pi / 2, 0, 0], name="Place" + tag)
            create_geometry_line(tree, [realize, fill, place], ins=curves.geometry_out)
            entries += [realize, fill, place]
            ends.append(place)

        pair = JoinGeometry(tree, location=(26.4, y - 1.6))
        for end in ends:
            tree.links.new(end.geometry_out, pair.geometry_in)
        # the plain characters in GlyphColor, the instructions in their own -
        # the same test and the same palette as everywhere else in the video
        which = instruction_selector(tree, character.std_out, self.opcode_colors,
                                     location=(26.4, y - 3.4),
                                     name="AsciiColorSelector")
        painters = [SetMaterial(tree, location=(27.4, y - 1.6),
                                material=dials.out("GlyphColor"),
                                name="PaintAsciiPlain")]
        painters += [SetMaterial(tree, location=(27.4, y - 2.0 - 0.3 * row),
                                 selection=which.outputs[node_name],
                                 material=dials.out(node_name),
                                 name="PaintAscii" + node_name, hide=True)
                     for row, (node_name, _, _) in enumerate(self.opcode_colors)]
        create_geometry_line(tree, [pair] + painters)

        grown = JoinGeometry(tree, location=(29.4, y - 1.6))
        tree.links.new(painters[-1].geometry_out, grown.geometry_in)
        tree.links.new(zone.repeat_input.outputs["Geometry"], grown.geometry_in)
        tree.links.new(grown.geometry_out, zone.repeat_output.inputs["Geometry"])

        box = self._create_table_frame(tree, control, zone.geometry_out,
                                       location=(31, y - 0.6),
                                       label="AsciiTableFrame")
        joined = JoinGeometry(tree, location=(40, y))
        tree.links.new(zone.geometry_out, joined.geometry_in)
        tree.links.new(box, joined.geometry_in)

        # ... and the whole of it comes in from the left as the old one leaves
        coming = make_function(
            tree, name="TableArriving", location=(40, y - 2.4), hide=True,
            functions={"In": ["gone,1,-,%s,*" % self.TABLE_SLIDE, "0", "0"]},
            inputs=["gone"], outputs=["In"], scalars=["gone"], vectors=["In"])
        tree.links.new(swap, coming.inputs["gone"])
        slid = TransformGeometry(tree, location=(41.4, y), geometry=joined.geometry_out,
                                 translation=coming.outputs["In"],
                                 name="SlideAsciiTable")

        frame = Frame(tree, location=(16.6, y + 1.0), label="AsciiTable")
        frame.add([dials, table, commands, zone, places, character, big,
                   char_curves, code, label, code_curves, pair, which, grown,
                   joined, coming, slid] + entries + painters)
        return slid.geometry_out

    # ----------------------------------------------------------------
    def _letters_landed(self, tree, control):
        """``LetterTransfer``: how many letters have reached the tape by now.

        Zero until ``SwitchLetterTime``, one more every ``LetterDuration``
        after it, and never more than the program is long. Two frames ask for
        it - the tape, to know what to write on a cell, and the copy of the
        program, to know which of its columns have left - so it is built once
        and handed to both.

        :return: an INT socket, the number of letters that have landed.
        """
        if self._landed is not None:
            return self._landed
        dials = self._unpack(tree, control, "SwitchLetterTime", "LetterDuration",
                             location=(17, -31.4), name="TransferControl")
        now = SceneTime(tree, location=(17, -32.4), std_out="Seconds",
                        name="TransferClock")
        # +1 so that the first letter lands *at* SwitchLetterTime rather than
        # one LetterDuration after it
        count = make_function(
            tree, name="LettersLanded", location=(18.6, -31.8), hide=False,
            functions={"Landed": "seconds,start,-,duration,/,floor,1,+,0,max,%d,min"
                                 % len(self.program)},
            inputs=["seconds", "start", "duration"], outputs=["Landed"],
            scalars=["seconds", "start", "duration"], integers=["Landed"])
        for socket, socket_name in ((now.std_out, "seconds"),
                                    (dials.out("SwitchLetterTime"), "start"),
                                    (dials.out("LetterDuration"), "duration")):
            tree.links.new(socket, count.inputs[socket_name])

        frame = Frame(tree, location=(16.6, -30.8), label="LetterTransfer")
        frame.add([dials, now, count])
        self._landed = count.outputs["Landed"]
        return self._landed

    # ----------------------------------------------------------------
    def _program_wipe(self, tree, control):
        """``ProgramWipe``: the edge that takes the old program away.

        One number, shared by everything the wipe removes: the x the wipe has
        reached. It starts a margin left of the input display at
        ``ProgramDisappearTime`` and arrives a margin past its right edge
        ``ProgramDisappearDuration`` later, and whatever stands to the left of
        it is gone. So the program and the box it is written in are taken away
        the way a line of text is read, rather than fading out as one thing -
        the copy above the tape is by then the only program left.

        The edge is worked out from the display rather than from a bounding
        box of what is being deleted: the box would shrink as the wipe ate
        into it and the wipe would run away with itself.

        :return: a FLOAT socket, the x the wipe has reached.
        """
        if self._wipe_edge is not None:
            return self._wipe_edge
        dials = self._unpack(tree, control, "InputPosition", "InputDisplaySize",
                             "ProgramDisappearTime", "ProgramDisappearDuration",
                             location=(31, -17.8), name="WipeControl")
        now = SceneTime(tree, location=(31, -19.0), std_out="Seconds",
                        name="WipeClock")
        edge = make_function(
            tree, name="WipeEdge", location=(32.6, -18.2), hide=False,
            aux_functions={"gone": "seconds,start,-,duration,/,0,max,1,min"},
            functions={"Edge": "place_x,width,0.5,*,-,%s,-,gone,width,%s,+,*,+"
                               % (self.WIPE_MARGIN, 2 * self.WIPE_MARGIN)},
            inputs=["place", "width", "seconds", "start", "duration"],
            outputs=["Edge"], vectors=["place"],
            scalars=["width", "seconds", "start", "duration", "gone", "Edge"])
        for socket, socket_name in (
                (dials.out("InputPosition"), "place"),
                (dials.out("InputDisplaySize"), "width"),
                (now.std_out, "seconds"),
                (dials.out("ProgramDisappearTime"), "start"),
                (dials.out("ProgramDisappearDuration"), "duration")):
            tree.links.new(socket, edge.inputs[socket_name])

        self._wipe_frame = Frame(tree, location=(30.6, -17.0), label="ProgramWipe")
        self._wipe_frame.add([dials, now, edge])
        self._wipe_edge = edge.outputs["Edge"]
        return self._wipe_edge

    # ----------------------------------------------------------------
    def _wipe(self, tree, control, geometry, location, name):
        """*geometry*, with whatever the wipe has passed over deleted.

        A point is gone once the edge of :meth:`_program_wipe` has passed its
        x, and ``Delete Geometry`` in ALL mode takes the faces hanging off it
        with it - so a letter of the program leaves as a letter rather than
        losing its vertices one at a time.

        :return: the geometry socket of what is left.
        """
        edge = self._program_wipe(tree, control)
        x, y = location
        here = Position(tree, location=(x, y - 0.8), name=name + "Here")
        passed = make_function(
            tree, name=name + "Passed", location=(x + 1, y - 0.8), hide=True,
            functions={"Gone": "here_x,edge,<"},
            inputs=["here", "edge"], outputs=["Gone"],
            vectors=["here"], scalars=["edge"], booleans=["Gone"])
        tree.links.new(here.std_out, passed.inputs["here"])
        tree.links.new(edge, passed.inputs["edge"])
        gone = DeleteGeometry(tree, location=(x + 2, y), domain="POINT", mode="ALL",
                              geometry=geometry, selection=passed.outputs["Gone"],
                              name=name)
        self._wipe_frame.add([here, passed, gone])
        return gone.geometry_out

    # ----------------------------------------------------------------
    def _output_string(self, tree, control, text):
        """What the output display reads: nothing, once the tape has it.

        The box hands the word over to :meth:`_create_output_copy_frame` the
        moment it sets off, and that copy is what is seen crossing the
        picture. The letters of it are laid out from the same measurements as
        this text, so the handover is a swap of two drawings that are in the
        same place, and nothing moves at the moment it happens.
        """
        now = SceneTime(tree, location=(21.4, -8.4), std_out="Seconds",
                        name="HandoverClock")
        dials = self._unpack(tree, control, "OutputDisappearTime",
                             location=(21.4, -9.2), name="HandoverControl")
        arrived = make_function(
            tree, name="OutputHasArrived", location=(22.8, -8.7), hide=True,
            functions={"There": "seconds,start,<,not"},
            inputs=["seconds", "start"], outputs=["There"],
            scalars=["seconds", "start"], booleans=["There"])
        for socket, socket_name in (
                (now.std_out, "seconds"),
                (dials.out("OutputDisappearTime"), "start")):
            tree.links.new(socket, arrived.inputs[socket_name])
        left = Switch(tree, location=(24.0, -8.4), input_type="STRING",
                      switch=arrived.outputs["There"], false=text, true="",
                      name="OutputInTheBox")
        frame = Frame(tree, location=(21.0, -7.8), label="OutputHandover")
        frame.add([now, dials, arrived, left])
        return left.std_out

    # ----------------------------------------------------------------
    def _create_display_frame(self, tree, control, label, width, position, location,
                              control_at):
        """The read-outs. The two of them lead very different lives here."""
        if label == "OutputDisplay":
            return self._create_output_display_frame(tree, control, width, position)
        box = super()._create_display_frame(tree, control, label, width, position,
                                            location, control_at)
        if label != "InputDisplay":
            return box
        return self._wipe(tree, control, box, location=(34, -19.6),
                          name="WipeInputDisplay")

    # ----------------------------------------------------------------
    def _create_output_display_frame(self, tree, control, width, position):
        """``OutputDisplay``: the box the machine printed into, and its arrow.

        Once the printed string is on the tape the box has nothing left to
        say, so it stops being a read-out and becomes a pointer: it turns
        ``ArrowColor`` at ``OutputRecolorTime`` and then, over
        ``OutputMorphDuration`` from ``OutputMorphTime``, unrolls into an
        arrow standing over the cell that holds the last letter it printed -
        the ``O`` of ``HELLO``.

        **Why the box is built differently here.** The simple machine draws
        it with ``Curve Wireframe``, which is a mesh and has nothing a morph
        can hold on to. :class:`~geometry_nodes.nodes.MorphNode2` wants both
        shapes as *a curve carrying a radius*, so that a blend of the two
        paths and the two radius profiles sweeps out the whole way between
        them - see the ``morphing`` scene and
        :class:`~geometry_nodes.modifier_video_brainfuck.TubeMorphModifier`,
        which is where this is taken from. So the rectangle is written that
        way from the first frame of the shot rather than swapped for one when
        the morph starts: at ``Morph Parameter`` 0 it is the same box, drawn
        as a swept tube instead of a wireframe.

        The frame is cut open to be a curve with two ends, which leaves a
        notch one sample wide in it - :attr:`ARROW_SAMPLES` is high enough
        that it reads as a mitre on a corner rather than a gap. ``Close
        Loop`` would sweep it shut and put a bridge across the arrow instead,
        which is the more visible of the two.

        :return: the geometry socket of the box, whatever it is by then.
        """
        y = -41.0
        dials = self._unpack(tree, control, width, position, "TapePosition",
                             "TapeSize", "CellSize", "OutputRecolorTime",
                             "OutputMorphTime", "OutputMorphDuration",
                             "FrameColor", "ArrowColor",
                             location=(26, y), name="OutputDisplayControl")

        # --- curve 1: the box the machine printed into ------------------
        # Not a ``Quadrilateral``: that is a *cyclic* curve, and a cyclic
        # curve resampled to N points and then swept as an open one is one
        # segment short - the gap where its two ends should meet. This is the
        # same rectangle written as five corners, the first of them repeated
        # at the end, so the sweep goes the whole way round and closes on
        # itself with the two flat caps touching.
        step = MeshLine(tree, location=(27, y - 0.4), mode="OFFSET", count=5,
                        start_location=Vector(), offset=Vector([1, 0, 0]),
                        name="OutputCornerPoints")
        which = Index(tree, location=(27, y - 1.4), name="WhichCorner")
        corners = make_function(
            tree, name="OutputCorners", location=(28, y - 1.4), hide=True,
            # only one of the two tests can hold, so their sum is 0 or 1 and
            # this is -1 on the left and bottom, +1 on the right and top
            aux_functions={"right": "i,1,=,i,2,=,+,2,*,1,-",
                           "top": "i,2,=,i,3,=,+,2,*,1,-"},
            functions={"At": ["right,width,*,0.5,*",
                              "top,%s,*,0.5,*" % self.display_height, "0"]},
            inputs=["i", "width"], outputs=["At"],
            integers=["i"], scalars=["width", "right", "top"], vectors=["At"])
        tree.links.new(which.std_out, corners.inputs["i"])
        tree.links.new(dials.out(width), corners.inputs["width"])
        box = SetPosition(tree, location=(28, y - 0.4), geometry=step.geometry_out,
                          position=corners.outputs["At"], name="OutputRectangle")
        loop = MeshToCurve(tree, location=(29, y - 0.4), mesh=box.geometry_out,
                           name="OutputOutline")
        thick = SetCurveRadius(tree, location=(30.4, y - 0.4),
                               curve=loop.geometry_out, radius=self.frame_radius,
                               name="OutputThickness")
        stood = TransformGeometry(tree, location=(30, y - 0.4),
                                  geometry=thick.geometry_out,
                                  translation=dials.out(position),
                                  rotation=[pi / 2, 0, 0],
                                  name="PlaceOutputDisplay")

        # --- curve 2: the arrow, as an axis and a radius ----------------
        # the cell it stands over is where the tape's Mesh Line puts it: the
        # points are spread between the two ends of a line TapeSize*CellSize
        # long, so they sit one TapeSize/(TapeSize-1) cell apart, not one cell
        ends = make_function(
            tree, name="ArrowEnds", location=(28, y - 2.0), hide=False,
            aux_functions={"across": "place_x,%d,size,*,cell,*,size,1,-,/,+"
                                     % self.arrow_cell},
            functions={"Tip": ["across", "place_y", "place_z,%s,+" % self.ARROW_GAP],
                       "Foot": ["across", "place_y", "place_z,%s,+"
                                % (self.ARROW_GAP + self.ARROW_SHAFT_LENGTH
                                   + self.ARROW_HEAD_LENGTH)]},
            inputs=["place", "size", "cell"], outputs=["Tip", "Foot"],
            vectors=["place", "Tip", "Foot"], integers=["size"],
            scalars=["cell", "across"])
        # the tape steps down at the end of the shot and the arrow points at
        # a cell of it, so it reads the same drifted position the cells do
        stand, drift = self._tape_stand(tree, control, dials.out("TapePosition"),
                                        location=(27, y - 3.0), name="ArrowStand")
        for socket, socket_name in ((stand, "place"),
                                    (dials.out("TapeSize"), "size"),
                                    (dials.out("CellSize"), "cell")):
            tree.links.new(socket, ends.inputs[socket_name])
        axis = CurveLine(tree, location=(30, y - 2.0), mode="POINTS",
                         start=ends.outputs["Foot"], end=ends.outputs["Tip"],
                         name="ArrowAxis")
        sampled = ResampleCurve(tree, location=(31, y - 2.0), mode="Count",
                                curve=axis.geometry_out, count=self.ARROW_SAMPLES,
                                name="ArrowSamples")
        # the radius is what makes the axis an arrow: it holds at the shaft,
        # steps out to the barbs and falls to nothing at the point
        along = SplineParameter(tree, location=(30, y - 3.2), std_out="Factor",
                                name="AlongArrow")
        profile = make_function(
            tree, name="ArrowRadius", location=(31, y - 3.2), hide=True,
            aux_functions={
                "shaft": "u,%.6f,<" % self.barb_fraction,
                "head": "1,u,-,%.6f,/,%.6f,*" % (1 - self.barb_fraction,
                                                 self.ARROW_HEAD_RADIUS)},
            functions={"radius": "shaft,%.6f,*,1,shaft,-,head,*,+"
                                 % self.ARROW_SHAFT_RADIUS},
            inputs=["u"], outputs=["radius"],
            scalars=["u", "shaft", "head", "radius"])
        tree.links.new(along.std_out, profile.inputs["u"])
        shaped = SetCurveRadius(tree, location=(32, y - 2.0),
                                curve=sampled.geometry_out,
                                radius=profile.outputs["radius"],
                                name="ArrowProfile")

        # --- the morph, and the colour it happens in --------------------
        drive = make_function(
            tree, name="OutputMorphDriver", location=(31, y - 4.4), hide=True,
            functions={"Morph": "seconds,start,-,duration,/,0,max,1,min",
                       "Recolored": "seconds,recolor,<,not"},
            inputs=["seconds", "start", "duration", "recolor"],
            outputs=["Morph", "Recolored"],
            scalars=["seconds", "start", "duration", "recolor", "Morph"],
            booleans=["Recolored"])
        now = SceneTime(tree, location=(30, y - 4.4), std_out="Seconds",
                        name="MorphClock")
        for socket, socket_name in (
                (now.std_out, "seconds"),
                (dials.out("OutputMorphTime"), "start"),
                (dials.out("OutputMorphDuration"), "duration"),
                (dials.out("OutputRecolorTime"), "recolor")):
            tree.links.new(socket, drive.inputs[socket_name])
        morph = MorphNode2(tree, location=(33.4, y - 1.0),
                           curve1=stood.geometry_out, curve2=shaped.geometry_out,
                           morph_parameter=drive.outputs["Morph"],
                           samples=self.ARROW_SAMPLES,
                           profile_resolution=self.ARROW_PROFILE,
                           name="OutputMorph")
        material = Switch(tree, location=(33.4, y - 3.6), input_type="MATERIAL",
                          switch=drive.outputs["Recolored"],
                          false=dials.out("FrameColor"),
                          true=dials.out("ArrowColor"), name="OutputBoxColor")
        painted = SetMaterial(tree, location=(35, y - 1.0),
                              geometry=morph.geometry_out,
                              material=material.std_out, name="PaintOutputDisplay")

        frame = Frame(tree, location=(25.6, y + 0.8), label="OutputDisplay")
        frame.add([dials, step, which, corners, box, loop, thick, stood, ends,
                   axis, sampled, along, profile, shaped, now, drive, morph,
                   material, painted] + drift)
        return painted.geometry_out

    # ----------------------------------------------------------------
    def _create_program_strip(self, tree, control, variables, run):
        """The strip of the machine, wiped away when its time is up."""
        strip = super()._create_program_strip(tree, control, variables, run)
        return self._wipe(tree, control, strip, location=(34, -22.2),
                          name="WipeProgramStrip")

    def _tape_in_zone(self, tree, control, sim_in):
        """``Rebuild``: this frame's tape, at this frame's size.

        A ``Mesh Line`` of the current ``TapeSize`` and ``CellSize`` - the
        same line the ``Tape`` frame builds, but here it is rebuilt every
        frame rather than once - carrying the values of the tape of the frame
        before, read cell by cell with a ``Sample Index``.

        The guard is what makes a *growing* tape work. ``Sample Index`` clamps
        an index that is past the end of the geometry it is given, so without
        it every cell that has just appeared would come up holding a copy of
        what was in the last old cell instead of nothing.

        :return: the geometry the automaton reads and writes, and the nodes
            that build it, for the frame in the editor.
        """
        dials = self._unpack(tree, control, "TapeSize", "CellSize",
                             location=(2.2, 0.6), name="RebuildControl")
        old = sim_in.outputs["Geometry"]

        # the tape as it should look now
        end = make_function(
            tree, name="NewTapeEnd", location=(3.6, 0.6), hide=True,
            functions={"end": "e_x,size,cell,*,scale"},
            inputs=["size", "cell"], outputs=["end"],
            integers=["size"], scalars=["cell"], vectors=["end"])
        tree.links.new(dials.out("TapeSize"), end.inputs["size"])
        tree.links.new(dials.out("CellSize"), end.inputs["cell"])
        line = MeshLine(tree, location=(4.8, 0.8), mode="END_POINTS",
                        count=dials.out("TapeSize"),
                        start_location=Vector(), end_location=end.outputs["end"])

        # ... holding what the tape of the frame before held
        stored = NamedAttribute(tree, location=(2.2, -0.8), data_type="INT",
                                name="Value")
        here = Index(tree, location=(2.2, -1.8))
        was = SampleIndex(tree, location=(3.6, -0.8), data_type="INT",
                          domain="POINT", geometry=old, value=stored.std_out,
                          index=here.std_out, name="ValueBefore")
        size = DomainSize(tree, location=(2.2, -2.6), geometry=old,
                          component="MESH", name="TapeSizeBefore")
        existed = CompareNode(tree, location=(3.6, -2.6), operation="LESS_THAN",
                              data_type="INT", inputs0=here.std_out,
                              inputs1=size.node.outputs["Point Count"],
                              name="CellExisted", hide=True)
        value = Switch(tree, location=(4.8, -0.8), input_type="INT",
                       switch=existed.std_out, false=0, true=was.std_out,
                       name="CarriedValue")
        tape = StoredNamedAttribute(tree, location=(6.0, 0.8), data_type="INT",
                                    domain="POINT", name="Value",
                                    value=value.std_out, label="CarryValues")
        tree.links.new(line.geometry_out, tape.geometry_in)

        return tape.geometry_out, [dials, end, line, stored, here, was, size,
                                   existed, value, tape]

    # ----------------------------------------------------------------
    def _create_cell_values(self, tree, control, variables, run):
        """``CellValues``: what is written on a cell - a number, or a letter.

        The same numbers as the simple machine until the transfer reaches the
        cell, and the instruction the program holds at that position after it.
        The cell keeps whatever value it holds either way: this is what is
        *drawn* on it, and the tape reading as the program is the picture the
        shot is after.

        :return: the geometry socket of the lettering.
        """
        tape = run["Geometry"]
        dials = self._unpack(tree, control, "CellSize", "GlyphColor",
                             "OutputOffset", "OutputDisappearTime",
                             "OutputMoveDuration",
                             *[node_name for node_name, _, _ in self.opcode_colors],
                             location=(25, -1.4), name="ValuesControl")
        source = self._unpack(tree, variables, "Input", location=(25, -3.0),
                              name="ValuesVariables")
        landed = self._letters_landed(tree, control)

        value = NamedAttribute(tree, location=(26, -2), data_type="INT", name="Value")
        position = Position(tree, location=(26, -2.6))
        here = Index(tree, location=(26, -3.2))
        zone = ForEachZone(tree, location=(27, -1.4), domain="POINT", node_width=8,
                           geometry=tape)
        zone.add_socket(socket_type="INT", name="Value", value=value.std_out,
                        for_input=True)
        zone.add_socket(socket_type="VECTOR", name="Location", value=position.std_out,
                        for_input=True)
        # which cell this is, carried in rather than read inside: an Index in
        # the body of the zone is the index of the element being drawn, and
        # the cell is what the letter has to be chosen by
        zone.add_socket(socket_type="INT", name="Index", value=here.std_out,
                        for_input=True)
        column = zone.foreach_input.outputs["Index"]

        digits = ValueToString(tree, location=(28, -0.8), data_type="INT",
                               value=zone.foreach_input.outputs["Value"],
                               name="CellValue")
        letter = SliceString(tree, location=(29, -0.2), string=source.out("Input"),
                             position=column, length=1, name="CellLetter")
        written = CompareNode(tree, location=(28, -3.6), operation="LESS_THAN",
                              data_type="INT", inputs0=column, inputs1=landed,
                              name="LetterHasLanded", hide=True)
        text = Switch(tree, location=(29, -0.8), input_type="STRING",
                      switch=written.std_out, false=digits.std_out,
                      true=letter.std_out, name="CellText")

        # ... and what the machine printed, on the cells from OutputOffset on.
        # No copy of it travels the way the program's does: the string is one
        # word in a box and the box has somewhere else to be, so it is simply
        # here from OutputDisappearTime on and gone from the box at the same
        # moment - see :meth:`_output_string`.
        printed = run["Output"]
        spelt = StringLength(tree, location=(28, -4.2), string=printed,
                             name="PrintedLength")
        now = SceneTime(tree, location=(28, -4.8), std_out="Seconds",
                        name="OutputClock")
        holds = make_function(
            tree, name="CellHoldsOutput", location=(29, -4.4), hide=True,
            aux_functions={"place": "column,offset,-"},
            functions={"Shows": "seconds,start,duration,+,<,not,"
                                "place,0,<,not,and,place,spelt,<,and",
                       "Place": "place"},
            inputs=["column", "offset", "spelt", "seconds", "start", "duration"],
            outputs=["Shows", "Place"],
            integers=["column", "offset", "spelt", "place", "Place"],
            scalars=["seconds", "start", "duration"], booleans=["Shows"])
        for socket, socket_name in ((column, "column"),
                                    (dials.out("OutputOffset"), "offset"),
                                    (spelt.std_out, "spelt"),
                                    (now.std_out, "seconds"),
                                    (dials.out("OutputDisappearTime"), "start"),
                                    (dials.out("OutputMoveDuration"), "duration")):
            tree.links.new(socket, holds.inputs[socket_name])
        spelled = SliceString(tree, location=(30, -4.4), string=printed,
                              position=holds.outputs["Place"], length=1,
                              name="CellPrinted")
        text = Switch(tree, location=(31, -0.8), input_type="STRING",
                      switch=holds.outputs["Shows"], false=text.std_out,
                      true=spelled.std_out, name="CellTextOrOutput")
        size = MathNode(tree, location=(28, -2.4), operation="MULTIPLY",
                        inputs0=dials.out("CellSize"), inputs1=self.glyph_size,
                        name="NumberSize")
        curves = StringToCurves(tree, location=(32, -1.4), string=text.std_out,
                                size=size.std_out, align_x="CENTER", align_y="BOTTOM")
        realize = RealizeInstances(tree, location=(33, -1.4))
        fill = FillCurve(tree, location=(34, -1.4), mode="N-gons")
        painted = SetMaterial(tree, location=(35, -1.4),
                              material=dials.out("GlyphColor"), name="PaintNumber")
        # A letter that has landed keeps the colour it had in the program, so
        # that a "+" is the same colour wherever it is read - the strip it
        # came from, the copy it travelled in, and now the cell it sits on.
        # The same test decides it here as everywhere else, and it answers no
        # for a cell still showing a number and for the letters of HELLO, so
        # those stay in GlyphColor, which is the fall-back this chain starts
        # from.
        which = instruction_selector(tree, text.std_out, self.opcode_colors,
                                     location=(33, -3.6), name="CellColorSelector")
        painters = [SetMaterial(tree, location=(35, -3.6 - 0.3 * row),
                                selection=which.outputs[node_name],
                                material=dials.out(node_name),
                                name="PaintCell" + node_name, hide=True)
                    for row, (node_name, _, _) in enumerate(self.opcode_colors)]
        # the whole tape is laid back by tape_tilt further downstream, so a
        # number turned by the complement of that angle ends up standing
        # upright on a cell that is itself leaning away from the camera
        placed = TransformGeometry(tree, location=(37, -1.4),
                                   translation=zone.foreach_input.outputs["Location"],
                                   rotation=[pi / 2 - self.tape_tilt, 0, 0],
                                   name="PlaceNumber")
        zone.create_geometry_line([realize, fill, painted] + painters + [placed],
                                  ins=curves.geometry_out)

        frame = Frame(tree, location=(25.6, -0.6), label="CellValues")
        frame.add([dials, source, value, position, here, zone, digits, letter,
                   written, spelt, now, holds, spelled, text, size, curves,
                   realize, fill, painted, which, placed] + painters)
        return zone.geometry_out

    # ----------------------------------------------------------------
    def _extra_geometry(self, tree, control, variables, run):
        """The two travelling copies, on top of everything the machine draws."""
        return [self._create_program_copy_frame(tree, control, variables),
                self._create_output_copy_frame(tree, control, run),
                self._create_ascii_table_frame(tree, control)]

    # ----------------------------------------------------------------
    def _create_output_copy_frame(self, tree, control, run):
        """``OutputCopy``: the printed word on its way from the box to the tape.

        ``HELLO`` does not blink out of the read-out and reappear on the
        cells; it is carried there, one letter per cell, shrinking on the way
        from the size of a read-out to the size of a letter on a cell. The
        letters spread apart as they go - they start at the pitch the font set
        them in and land at the pitch of the tape - which is what makes the
        shot read as *the printed output becoming the data*.

        **The letters are not laid out again, they are picked up.** ``String
        to Curves`` hands out one *instance per character*, already standing
        where the font wants it, and ``Index`` on that domain is the position
        of the character in the string. So this is the read-out's own text
        with each of its letters told how far to move, which is why the moment
        the copy takes over from the read-out cannot be seen. Laying the
        letters out again from measurements of their own outlines is what does
        not work: ink is not advance, and the word comes apart by a few
        percent a letter.

        Each letter is sent to cell ``OutputOffset + i``, at the tape's own
        point spacing of ``TapeSize/(TapeSize-1)`` cells and half a glyph
        above it, which is where :meth:`_create_cell_values` is about to draw
        it. The offsets are in the text's own frame, which the transform at
        the end stands up into the shot - so its x is the shot's x and its y
        the shot's z.

        :return: the geometry socket of the letters in flight.
        """
        y = -47.0
        printed = run["Output"]
        dials = self._unpack(tree, control, "OutputPosition", "OutputOffset",
                             "OutputDisappearTime", "OutputMoveDuration",
                             "TapePosition", "TapeSize", "CellSize", "GlyphColor",
                             location=(20, y), name="OutputCopyControl")
        now = SceneTime(tree, location=(20, y - 1.8), std_out="Seconds",
                        name="OutputCopyClock")
        box_size = 0.6 * self.display_height

        # the read-out's own drawing of the word, one instance per letter
        letters = StringToCurves(tree, location=(22, y), string=printed,
                                 size=box_size, align_x="CENTER",
                                 align_y="MIDDLE", hide=True, name="WordInFlight")
        which = Index(tree, location=(22, y - 1.0), name="WhichLetter")
        here = Position(tree, location=(22, y - 1.6), name="LetterHere")

        flight = make_function(
            tree, name="LetterFlight", location=(23.6, y - 1.2), hide=False,
            aux_functions={
                "gone": "seconds,start,-,duration,/,0,max,1,min",
                "step": "size,cell,*,size,1,-,/",
                "to_x": "place_x,offset,i,+,step,*,+,home_x,-",
                "to_y": "place_z,cell,%s,*,2,/,+,home_z,-" % self.glyph_size},
            functions={"Offset": ["gone,to_x,here_x,-,*",
                                  "gone,to_y,here_y,-,*", "0"],
                       "Shrink": "1,gone,cell,%s,*,%s,/,1,-,*,+"
                                 % (self.glyph_size, box_size),
                       # before it sets off the read-out has the word and
                       # after it lands the tape does; either way not this
                       "Grounded": "seconds,start,<,not,gone,1,<,and,not"},
            inputs=["here", "home", "place", "size", "cell", "offset", "i",
                    "seconds", "start", "duration"],
            outputs=["Offset", "Shrink", "Grounded"],
            vectors=["here", "home", "place", "Offset"],
            integers=["size", "offset", "i"],
            scalars=["cell", "seconds", "start", "duration", "gone", "step",
                     "to_x", "to_y", "Shrink"],
            booleans=["Grounded"])
        for socket, socket_name in (
                (here.std_out, "here"),
                (dials.out("OutputPosition"), "home"),
                (dials.out("TapePosition"), "place"),
                (dials.out("TapeSize"), "size"),
                (dials.out("CellSize"), "cell"),
                (dials.out("OutputOffset"), "offset"),
                (which.std_out, "i"),
                (now.std_out, "seconds"),
                (dials.out("OutputDisappearTime"), "start"),
                (dials.out("OutputMoveDuration"), "duration")):
            tree.links.new(socket, flight.inputs[socket_name])

        # TranslateInstances takes an ``instances`` argument and does not
        # link it, so the geometry goes in by hand
        spread = TranslateInstances(tree, location=(25.4, y),
                                    translation=flight.outputs["Offset"],
                                    name="SpreadLetters")
        tree.links.new(letters.geometry_out, spread.geometry_in)
        shrunk = ScaleInstances(tree, location=(26.4, y),
                                geometry=spread.geometry_out,
                                scale=flight.outputs["Shrink"],
                                name="ShrinkLetters")
        realize = RealizeInstances(tree, location=(27.4, y))
        fill = FillCurve(tree, location=(28.4, y), mode="N-gons")
        painted = SetMaterial(tree, location=(29.4, y),
                              material=dials.out("GlyphColor"),
                              name="PaintLettersInFlight")
        put = TransformGeometry(tree, location=(30.4, y),
                                translation=dials.out("OutputPosition"),
                                rotation=[pi / 2, 0, 0],
                                name="PlaceLettersInFlight")
        gone = DeleteGeometry(tree, location=(31.4, y), domain="POINT", mode="ALL",
                              selection=flight.outputs["Grounded"],
                              name="OnlyWhileFlying")
        create_geometry_line(tree, [shrunk, realize, fill, painted, put, gone])

        frame = Frame(tree, location=(19.6, y + 1.0), label="OutputCopy")
        frame.add([dials, now, letters, which, here, flight, spread, shrunk,
                   realize, fill, painted, put, gone])
        return gone.geometry_out

    # ----------------------------------------------------------------
    def _create_program_copy_frame(self, tree, control, variables):
        """``CopyOfProgramStrip``: the program again, on its way to the tape.

        The program strip of :meth:`_create_program_strip` once more, with
        three differences, all of them because this one is going somewhere:

        - it is not a read-out of the machine, so nothing paints it by what
          has run and there is no box running along it. Every column keeps the
          colour of the instruction it is, which is what has to survive the
          journey onto the tape.
        - ``StripLayout2`` in place of ``StripLayout``: the same ruler with
          ``ShrinkFontSize`` and ``ShrinkSpacing`` folded into it, so the copy
          can be squeezed from the width of the input display down to the
          pitch of the cells without a node per multiplication.
        - it comes and goes. Before ``CopyProgramTime`` the whole thing is
          deleted rather than drawn somewhere out of shot, and each column is
          deleted again once its letter has landed on the tape.

        :return: the geometry socket of the copy.
        """
        y = -34.0
        source = self._unpack(tree, variables, "Input", location=(17, y),
                              name="CopyVariables")
        dials = self._unpack(
            tree, control, "InputPosition", "InputDisplaySize", "ShrinkFontSize",
            "ShrinkSpacing", "ProgramShift", "CopyProgramTime",
            *[node_name for node_name, _, _ in self.opcode_colors],
            location=(17, y - 0.8), name="CopyControl")
        landed = self._letters_landed(tree, control)
        program = source.out("Input")
        size = StringLength(tree, location=(18.6, y), string=program,
                            name="CopyLength")

        # the strip's own ruler, with the two shrink factors folded in: the
        # gap between columns is what the copy is squeezed by, and the letters
        # in it are shrunk by a factor of their own so that they can be made
        # to fit a cell without the columns having to
        ruler = make_function(
            tree, name="StripLayout2", location=(20.0, y - 1.2), hide=False,
            aux_functions={"gap": "width,%d,/" % (len(self.program) + 2)},
            functions={"First": "place_x,width,0.5,*,-,gap,+",
                       "Spacing": "gap,space,*",
                       "Glyph": "gap,%s,*,font,*" % self.strip_glyph_size},
            inputs=["place", "width", "font", "space"],
            outputs=["First", "Spacing", "Glyph"],
            vectors=["place"],
            scalars=["width", "font", "space", "gap", "First", "Spacing", "Glyph"])
        for socket, socket_name in ((dials.out("InputPosition"), "place"),
                                    (dials.out("InputDisplaySize"), "width"),
                                    (dials.out("ShrinkFontSize"), "font"),
                                    (dials.out("ShrinkSpacing"), "space")):
            tree.links.new(socket, ruler.inputs[socket_name])

        # --- one column per instruction, as in the strip -----------------
        zone = RepeatZone(tree, location=(22, y), node_width=9,
                          iterations=size.std_out)
        column = zone.iteration
        letter = SliceString(tree, location=(23, y - 0.8), string=program,
                             position=column, length=1, name="CopyLetter")
        curves = StringToCurves(tree, location=(24, y - 0.8), string=letter.std_out,
                                size=ruler.outputs["Glyph"], align_x="CENTER",
                                align_y="MIDDLE", hide=True)
        realize = RealizeInstances(tree, location=(25, y - 0.8))
        fill = FillCurve(tree, location=(26, y - 0.8), mode="N-gons")
        at = make_function(
            tree, name="ColumnPlace2", location=(24, y + 0.6), hide=False,
            functions={"At": ["first,column,spacing,*,+", "place_y", "place_z"]},
            inputs=["first", "spacing", "column", "place"], outputs=["At"],
            scalars=["first", "spacing"], integers=["column"],
            vectors=["place", "At"])
        for socket, socket_name in ((ruler.outputs["First"], "first"),
                                    (ruler.outputs["Spacing"], "spacing"),
                                    (column, "column"),
                                    (dials.out("InputPosition"), "place")):
            tree.links.new(socket, at.inputs[socket_name])
        place = TransformGeometry(tree, location=(27, y - 0.8),
                                  translation=at.outputs["At"],
                                  rotation=[pi / 2, 0, 0], name="PlaceCopyColumn")

        which = instruction_selector(tree, letter.std_out, self.opcode_colors,
                                     location=(27, y - 2.6), name="CopyColorSelector")
        painters = [SetMaterial(tree, location=(28, y - 0.8 - 0.3 * row),
                                selection=which.outputs[node_name],
                                material=dials.out(node_name),
                                name="Copy" + node_name, hide=True)
                    for row, (node_name, _, _) in enumerate(self.opcode_colors)]
        create_geometry_line(tree, [realize, fill, place] + painters,
                             ins=curves.geometry_out)

        # a column that has been written onto the tape has left the copy
        gone = DeleteGeometry(tree, location=(29, y - 0.8), domain="POINT",
                              mode="ALL", geometry=painters[-1].geometry_out,
                              name="LetterHasLeft")
        left = CompareNode(tree, location=(28, y - 4.0), operation="LESS_THAN",
                           data_type="INT", inputs0=column, inputs1=landed,
                           name="ColumnHasLanded", hide=True)
        tree.links.new(left.std_out, gone.node.inputs["Selection"])
        grown = JoinGeometry(tree, location=(30, y - 0.8))
        tree.links.new(gone.geometry_out, grown.geometry_in)
        tree.links.new(zone.repeat_input.outputs["Geometry"], grown.geometry_in)
        tree.links.new(grown.geometry_out, zone.repeat_output.inputs["Geometry"])

        # --- and the copy as a whole ------------------------------------
        # deleted rather than moved out of shot until it is called for, so
        # that it costs nothing at all for the first seconds of the scene
        now = SceneTime(tree, location=(32, y - 2.0), std_out="Seconds",
                        name="CopyClock")
        early = CompareNode(tree, location=(33, y - 2.0), operation="LESS_THAN",
                            data_type="FLOAT", inputs0=now.std_out,
                            inputs1=dials.out("CopyProgramTime"),
                            name="BeforeTheCopy", hide=True)
        hidden = DeleteGeometry(tree, location=(33, y), domain="POINT", mode="ALL",
                                geometry=zone.geometry_out,
                                selection=early.std_out, name="NotYet")
        shifted = TransformGeometry(tree, location=(34, y),
                                    geometry=hidden.geometry_out,
                                    translation=dials.out("ProgramShift"),
                                    name="ShiftCopy")

        frame = Frame(tree, location=(16.6, y + 1.0), label="CopyOfProgramStrip")
        frame.add([source, dials, size, ruler, zone, letter, curves, realize, fill,
                   at, place, which, gone, left, grown, now, early, hidden,
                   shifted] + painters)
        return shifted.geometry_out


class BFFNode(NodeGroup):
    """One step of the BFF machine of the paper, as a node group.

    Hand it the tape and the three numbers that say where the machine is, and
    it hands back the tape with that one instruction carried out and the three
    numbers moved on. It holds no state of its own - the state lives in the
    simulation zone that drives it, which is what lets the same group run any
    number of machines from one node tree.

    **The interface.** Geometry in, geometry out, and the positions between:

    ``Geometry``
        the tape, one point per cell with the byte in the integer point
        attribute ``Value``. Both tapes of a pair are one geometry, joined end
        to end, and the node takes their length from the geometry itself - a
        ``Domain Size`` rather than a socket - so it runs a pair of 64 byte
        tapes, a single one, or eight of them without being told which.
    ``Head0``, ``Head1``
        the two data heads. ``<`` and ``>`` move head0, ``{`` and ``}`` move
        head1, and both run round memory as a ring.
    ``Counter``
        the program counter, an index into that same tape: the value of the
        cell it points at *is* the instruction, and anything that is not one of
        the ten is a no-op. That is what makes the tape a program.
    ``Fire``
        true on the one frame a step is due. Every instruction is ``AND``-ed
        with it, so an instruction takes effect once however many frames it is
        on screen for.
    ``Instruction``, ``Value0``, ``Value1``
        outputs, for read-outs: the opcode being executed and what the two
        heads are standing on. They are sampled inside anyway, and having them
        come out saves sampling the same three cells again to draw them.

    **Why the boundary is here.** The obvious interface - values in, values out
    - would leave two thirds of the machine outside: ``[`` and ``]`` find their
    partner by walking the tape (see :meth:`_bracket_scan`), and the two writes
    are ``Store Named Attribute`` with a *selection*, which is a field over the
    tape rather than a value. Both need the whole geometry, so the geometry is
    what crosses.
    """

    # ascii codes of the ten instructions
    DOT, PLUS, MINUS, LEFT, RIGHT = ord("."), ord("+"), ord("-"), ord("<"), ord(">")
    OPEN, CLOSE = ord("["), ord("]")
    BRACE_LEFT, BRACE_RIGHT = ord("{"), ord("}")
    COMMA = ord(",")
    #: the ten, as characters, for whoever has to colour or count them
    COMMANDS = ".,<>[]{}+-"

    def __init__(self, tree, **kwargs):
        self.name = get_from_kwargs(kwargs, "name", "BFFNode")
        super().__init__(
            tree,
            inputs={"Geometry": "GEOMETRY", "Head0": "INT", "Head1": "INT",
                    "Counter": "INT", "Fire": "BOOLEAN"},
            outputs={"Geometry": "GEOMETRY", "Head0": "INT", "Head1": "INT",
                     "Counter": "INT", "Instruction": "INT", "Value0": "INT",
                     "Value1": "INT"},
            # the nodes inside are placed by hand, in frames; the automatic
            # layout would throw that away
            auto_layout=False, name=self.name, **kwargs)
        self.geometry_in = self.node.inputs["Geometry"]
        self.geometry_out = self.node.outputs["Geometry"]

    # ----------------------------------------------------------------
    def fill_group_with_node(self, tree, **kwargs):
        """``Automaton``: what the ten instructions do.

        Every instruction is one ``Compare`` against its ascii code, ``AND``-ed
        with ``Fire`` - "a new step has just begun".

        Which head each instruction moves is the whole of the difference
        between this machine and a one-headed brainfuck:

        ``<`` ``>``
            move head0, the head ``+``, ``-``, ``[`` and ``]`` work on.
        ``{`` ``}``
            move head1, which has no arithmetic of its own - it is a place to
            copy to and from, nothing more.
        ``.`` ``,``
            do not print. ``.`` copies the cell under head0 into the cell under
            head1, and ``,`` copies it back the other way. That is the whole of
            the machine's data movement, and it is what lets a program on the
            tape write another program.
        """
        ins, outs = self.group_inputs, self.group_outputs
        ins.location = (10.4 * 200, 2 * 100)
        outs.location = (28.0 * 200, 2 * 100)

        geometry = ins.outputs["Geometry"]
        head = ins.outputs["Head0"]
        mate = ins.outputs["Head1"]
        counter = ins.outputs["Counter"]
        fire = ins.outputs["Fire"]

        built = []

        def keep(node):
            built.append(node)
            return node

        # the instruction is the value of the cell the counter points at
        held = keep(NamedAttribute(tree, location=(11.2, 5.4), data_type="INT",
                                   name="Value"))
        instruction = keep(SampleIndex(tree, location=(12.4, 5.4), data_type="INT",
                                       domain="POINT", geometry=geometry,
                                       value=held.std_out, index=counter,
                                       name="Instruction"))
        opcode = instruction.std_out
        # how far the ring of memory reaches, read off the tape rather than
        # passed in: whatever geometry the node is handed is the whole of
        # memory, however many tapes were joined to make it
        size = keep(DomainSize(tree, location=(11.2, 6.2), component="MESH",
                               geometry=geometry, name="MemorySize"))
        memory = size.outputs["Point Count"]

        def decodes(code, row, label):
            """``True`` on the frame the instruction ``code`` is executed."""
            is_op = keep(CompareNode(tree, location=(12.4, row), operation="EQUAL",
                                     data_type="INT", inputs0=opcode, inputs1=code,
                                     name="Is" + label, hide=True))
            return keep(BooleanMath(tree, location=(13.6, row), operation="AND",
                                    inputs0=is_op.std_out, inputs1=fire,
                                    name="Do" + label, hide=True)).std_out

        def step_of(condition, row, label):
            """1 on the frame the instruction runs, 0 otherwise."""
            return keep(Switch(tree, location=(14.8, row), input_type="INT",
                               switch=condition, false=0, true=1, name=label)).std_out

        # Memory is a *ring*: a head that walks left off cell 0 comes back on at
        # the last cell, and one that walks right off the last cell comes back
        # on at cell 0. Where two tapes are joined end to end that is also what
        # carries a head from the first onto the second - cell 63 and cell 64
        # are neighbours. The counter is the one thing that does not wrap: it
        # runs off the end and the machine halts, as in the paper's interpreter.

        def walk(where, forward, backward, row, label):
            """A head moved one cell, round the ring of memory."""
            ahead = keep(IntegerMath(tree, location=(16.0, row), operation="ADD",
                                     inputs0=where, inputs1=forward,
                                     name=label + "Right"))
            back = keep(IntegerMath(tree, location=(17.0, row), operation="SUBTRACT",
                                    inputs0=ahead.std_out, inputs1=backward,
                                    name=label + "Left"))
            # FLOORED_MODULO, not MODULO: blender's plain modulo takes the sign
            # of the dividend, so -1 would stay -1 and the head would walk off
            # the tape rather than round it
            return keep(IntegerMath(tree, location=(18.0, row),
                                    operation="FLOORED_MODULO",
                                    inputs0=back.std_out, inputs1=memory,
                                    name=label + "RoundTheRing")).std_out

        # --- the two heads ----------------------------------------------
        pointer = walk(head,
                       step_of(decodes(self.RIGHT, 4.2, "Right"), 4.2, "StepRight"),
                       step_of(decodes(self.LEFT, 3.4, "Left"), 3.4, "StepLeft"),
                       3.8, "Head0")
        partner = walk(mate,
                       step_of(decodes(self.BRACE_RIGHT, 2.6, "BraceRight"), 2.6,
                               "SlideRight"),
                       step_of(decodes(self.BRACE_LEFT, 1.8, "BraceLeft"), 1.8,
                               "SlideLeft"),
                       2.2, "Head1")

        # --- what the two cells hold ------------------------------------
        # this is where the values live: an integer attribute of the tape
        # geometry, which the simulation zone hands from frame to frame
        cell = SampleIndex(tree, location=(13.6, 1.0), data_type="INT", domain="POINT",
                           geometry=geometry, value=held.std_out,
                           index=head, name="CellUnderHead0")
        other = SampleIndex(tree, location=(13.6, 0.0), data_type="INT", domain="POINT",
                            geometry=geometry, value=held.std_out,
                            index=mate, name="CellUnderHead1")

        plus = step_of(decodes(self.PLUS, -0.8, "Plus"), -0.8, "Increment")
        minus = step_of(decodes(self.MINUS, -1.6, "Minus"), -1.6, "Decrement")
        raised = IntegerMath(tree, location=(16.0, 0.6), operation="ADD",
                             inputs0=cell.std_out, inputs1=plus, name="CellPlus")
        lowered = IntegerMath(tree, location=(17.0, 0.6), operation="SUBTRACT",
                              inputs0=raised.std_out, inputs1=minus, name="CellMinus")
        # a cell holds a byte: "-" on a 0 leaves 255 and "+" on a 255 leaves 0,
        # which is what keeps every value a character of the code table and
        # what the paper's interpreter does with its & 0xFF
        byte = IntegerMath(tree, location=(17.6, 0.6), operation="FLOORED_MODULO",
                           inputs0=lowered.std_out, inputs1=256, name="AsAByte")
        # "," overrides the arithmetic rather than adding to it: it is the one
        # instruction that puts something into the cell under head0 from outside
        reads = decodes(self.COMMA, -2.4, "Comma")
        writes = decodes(self.DOT, -3.2, "Dot")
        fetched = Switch(tree, location=(18.0, 0.6), input_type="INT", switch=reads,
                         false=byte.std_out, true=other.std_out,
                         name="NewHead0Value")

        # only the cell a head is on is written, every other one keeps what it
        # had - this selection is the whole of "+" and "-"
        here = Index(tree, location=(16.0, -0.4))
        on_head = CompareNode(tree, location=(17.0, -0.4), operation="EQUAL",
                              data_type="INT", inputs0=here.std_out, inputs1=head,
                              name="AtHead0", hide=True)
        write_head = StoredNamedAttribute(tree, location=(19.4, 0.6), data_type="INT",
                                          domain="POINT", name="Value",
                                          selection=on_head.std_out,
                                          value=fetched.std_out, label="WriteHead0")
        tree.links.new(geometry, write_head.geometry_in)

        # The cell under head1 is only ever written by ".", and the selection
        # says so rather than the value: without the "AND" a "+" on a frame
        # where both heads are on the same cell would be undone by this node
        # writing the value it sampled before the increment.
        on_mate = CompareNode(tree, location=(17.0, -1.2), operation="EQUAL",
                              data_type="INT", inputs0=here.std_out, inputs1=mate,
                              name="AtHead1", hide=True)
        copies = BooleanMath(tree, location=(18.0, -1.2), operation="AND",
                             inputs0=on_mate.std_out, inputs1=writes,
                             name="CopyToHead1", hide=True)
        write_mate = StoredNamedAttribute(tree, location=(20.4, 0.6), data_type="INT",
                                          domain="POINT", name="Value",
                                          selection=copies.std_out,
                                          value=cell.std_out, label="WriteHead1")
        create_geometry_line(tree, [write_head, write_mate])

        # --- the loop, and where the counter goes next -------------------
        # "[" and "]" are the same instruction read in opposite directions:
        # each looks at the cell under head0 and either falls through to the
        # next instruction or jumps to its partner. "[" leaves the loop when
        # the cell has run down to zero, "]" goes round again while it has not.
        empty = CompareNode(tree, location=(14.8, -4.0), operation="EQUAL",
                            data_type="INT", inputs0=cell.std_out, inputs1=0,
                            name="CellIsEmpty", hide=True)
        filled = CompareNode(tree, location=(14.8, -4.8), operation="NOT_EQUAL",
                             data_type="INT", inputs0=cell.std_out, inputs1=0,
                             name="CellIsNotEmpty", hide=True)
        skips = BooleanMath(tree, location=(16.0, -4.0), operation="AND",
                            inputs0=decodes(self.OPEN, -4.0, "Open"),
                            inputs1=empty.std_out, name="SkipLoop", hide=True)
        repeats = BooleanMath(tree, location=(16.0, -4.8), operation="AND",
                              inputs0=decodes(self.CLOSE, -4.8, "Close"),
                              inputs1=filled.std_out, name="RepeatLoop", hide=True)
        jumping = BooleanMath(tree, location=(17.0, -4.4), operation="OR",
                              inputs0=skips.std_out, inputs1=repeats.std_out,
                              name="TakeJump", hide=True)
        target = self._bracket_scan(tree, geometry, opcode, counter, memory)
        onward = IntegerMath(tree, location=(14.8, -6.6), operation="ADD",
                             inputs0=counter, inputs1=1, name="NextInstruction")
        jumped = Switch(tree, location=(18.0, -5.8), input_type="INT",
                        switch=jumping.std_out, false=onward.std_out,
                        true=target, name="CounterAfterStep")
        # on the frames in between two steps, and after the program has ended,
        # the counter stays where it is
        moved = Switch(tree, location=(19.0, -5.8), input_type="INT",
                       switch=fire, false=counter, true=jumped.std_out,
                       name="NewCounter")

        for socket, name in ((write_mate.geometry_out, "Geometry"),
                             (pointer, "Head0"), (partner, "Head1"),
                             (moved.std_out, "Counter"),
                             (opcode, "Instruction"), (cell.std_out, "Value0"),
                             (other.std_out, "Value1")):
            tree.links.new(socket, outs.inputs[name])

        frame = Frame(tree, location=(11.0, 7.0), label="Automaton")
        frame.add(built + [cell, other, raised, lowered, byte, fetched, here,
                           on_head, write_head, on_mate, copies, write_mate,
                           empty, filled, skips, repeats, jumping,
                           onward, jumped, moved])

    # ----------------------------------------------------------------
    def _bracket_scan(self, tree, geometry, opcode, counter, memory):
        """``FindTheBracket``: where a jump taken at the counter would go.

        The partner of a bracket is *searched for*, once per frame, in the tape
        as it stands. That is the price of the program being the tape: a jump
        table worked out in python when the graph is built - which is what a
        one-headed brainfuck can afford - would be a table for a program that
        the very next ``.`` may overwrite, and a soup of random bytes has
        unmatched brackets in it anyway, which no table can describe.

        The search is the one in ``brainfuck/bff/bff.py``, laid out as a repeat
        zone. It starts one cell beyond the bracket with a depth of 1 and walks
        until the depth reaches zero or it leaves memory, counting a bracket of
        the same kind as one deeper and one of the other kind as one shallower.
        One zone serves both directions: ``[`` searches forwards and ``]``
        backwards, and the *sign* of the step is all that differs, because the
        two brackets swap roles along with it.

        :param geometry: the tape
        :param opcode: the instruction under the counter
        :param counter: the program counter
        :param memory: how many cells there are altogether
        :return: the socket holding the counter's new value if the jump is
            taken - one past the partner, or out of memory when there is none,
            which halts the machine exactly as the paper's interpreter does.
        """
        pieces = []

        def keep(node):
            pieces.append(node)
            return node

        opens = keep(CompareNode(tree, location=(12.4, -8.0), operation="EQUAL",
                                 data_type="INT", inputs0=opcode, inputs1=self.OPEN,
                                 name="ScanForwards", hide=True))
        step = keep(Switch(tree, location=(13.4, -8.0), input_type="INT",
                           switch=opens.std_out, true=1, false=-1,
                           name="ScanDirection"))
        start = keep(IntegerMath(tree, location=(14.4, -8.0), operation="ADD",
                                 inputs0=counter, inputs1=step.std_out,
                                 name="ScanFrom"))

        # as many turns as there are cells: no partner can be further away than
        # the whole of memory, and a repeat zone cannot break out early anyway
        zone = RepeatZone(tree, location=(15.4, -8.0), node_width=8,
                          iterations=memory, geometry=geometry)
        for socket_name, value in (("Position", start.std_out), ("Depth", 1)):
            zone.add_socket(socket_type="INT", name=socket_name)
            if isinstance(value, int):
                zone.repeat_input.inputs[socket_name].default_value = value
            else:
                tree.links.new(value, zone.repeat_input.inputs[socket_name])
        at = zone.repeat_input.outputs["Position"]
        depth = zone.repeat_input.outputs["Depth"]

        x = 16.6
        held = keep(NamedAttribute(tree, location=(x, -9.4), data_type="INT",
                                   name="Value"))
        here = keep(SampleIndex(tree, location=(x, -8.6), data_type="INT",
                                domain="POINT",
                                geometry=zone.repeat_input.outputs["Geometry"],
                                value=held.std_out, index=at, name="ScannedCell"))
        is_open = keep(CompareNode(tree, location=(x + 1, -8.2), operation="EQUAL",
                                   data_type="INT", inputs0=here.std_out,
                                   inputs1=self.OPEN, name="ScanHitOpen", hide=True))
        is_close = keep(CompareNode(tree, location=(x + 1, -8.6), operation="EQUAL",
                                    data_type="INT", inputs0=here.std_out,
                                    inputs1=self.CLOSE, name="ScanHitClose", hide=True))
        deeper = keep(Switch(tree, location=(x + 2, -8.2), input_type="INT",
                             switch=is_open.std_out, true=1, false=0, name="Deeper"))
        shallower = keep(Switch(tree, location=(x + 2, -8.6), input_type="INT",
                                switch=is_close.std_out, true=1, false=0,
                                name="Shallower"))
        # +1 for a bracket of the kind that opened the search and -1 for its
        # partner, which the direction of the search turns round: going
        # backwards a "]" is one deeper and a "[" one shallower
        net = keep(IntegerMath(tree, location=(x + 3, -8.4), operation="SUBTRACT",
                               inputs0=deeper.std_out, inputs1=shallower.std_out,
                               name="NetDepth", hide=True))
        signed = keep(IntegerMath(tree, location=(x + 4, -8.4), operation="MULTIPLY",
                                  inputs0=net.std_out, inputs1=step.std_out,
                                  name="DepthChange", hide=True))
        # the zone runs its full count of turns whatever happens, so every turn
        # after the answer is found has to leave the state alone
        searching = keep(CompareNode(tree, location=(x + 1, -9.4),
                                     operation="GREATER_THAN", data_type="INT",
                                     inputs0=depth, inputs1=0,
                                     name="StillSearching", hide=True))
        after = keep(CompareNode(tree, location=(x + 1, -9.8), operation="GREATER_THAN",
                                 data_type="INT", inputs0=at, inputs1=-1,
                                 name="NotOffTheFront", hide=True))
        before = keep(CompareNode(tree, location=(x + 1, -10.2), operation="LESS_THAN",
                                  data_type="INT", inputs0=at, inputs1=memory,
                                  name="NotOffTheBack", hide=True))
        inside = keep(BooleanMath(tree, location=(x + 2, -10.0), operation="AND",
                                  inputs0=after.std_out, inputs1=before.std_out,
                                  name="InsideMemory", hide=True))
        active = keep(BooleanMath(tree, location=(x + 3, -9.6), operation="AND",
                                  inputs0=searching.std_out, inputs1=inside.std_out,
                                  name="KeepLooking", hide=True))
        sunk = keep(IntegerMath(tree, location=(x + 5, -8.4), operation="ADD",
                                inputs0=depth, inputs1=signed.std_out,
                                name="DepthAfter", hide=True))
        walked = keep(IntegerMath(tree, location=(x + 5, -9.0), operation="ADD",
                                  inputs0=at, inputs1=step.std_out,
                                  name="PositionAfter", hide=True))
        next_depth = keep(Switch(tree, location=(x + 6, -8.4), input_type="INT",
                                 switch=active.std_out, true=sunk.std_out,
                                 false=depth, name="Depth"))
        next_at = keep(Switch(tree, location=(x + 6, -9.0), input_type="INT",
                              switch=active.std_out, true=walked.std_out,
                              false=at, name="Position"))
        tree.links.new(next_depth.std_out, zone.repeat_output.inputs["Depth"])
        tree.links.new(next_at.std_out, zone.repeat_output.inputs["Position"])

        # the walk stops one past the partner, so a step back lands on it and
        # the counter goes on from the cell after that
        found = keep(CompareNode(tree, location=(24.4, -8.0), operation="EQUAL",
                                 data_type="INT",
                                 inputs0=zone.repeat_output.outputs["Depth"],
                                 inputs1=0, name="PartnerFound", hide=True))
        landed = keep(IntegerMath(tree, location=(24.4, -8.6), operation="SUBTRACT",
                                  inputs0=zone.repeat_output.outputs["Position"],
                                  inputs1=step.std_out, name="OnThePartner"))
        beyond = keep(IntegerMath(tree, location=(25.4, -8.6), operation="ADD",
                                  inputs0=landed.std_out, inputs1=1,
                                  name="PastThePartner"))
        # An unmatched bracket is the end of the program: the counter is sent
        # out of memory, where the machine's halting test stops it. Which way
        # out depends on which way the search went - past the last cell for a
        # "[" and before the first for a "]" - so that the counter reads
        # afterwards the way the paper's interpreter leaves it rather than
        # merely landing somewhere that happens to halt.
        past = keep(IntegerMath(tree, location=(25.4, -7.4), operation="ADD",
                                inputs0=memory, inputs1=1, name="PastTheEnd"))
        missing = keep(Switch(tree, location=(26.4, -7.4), input_type="INT",
                              switch=opens.std_out, true=past.std_out, false=-1,
                              name="NoPartner"))
        answer = keep(Switch(tree, location=(26.4, -8.0), input_type="INT",
                             switch=found.std_out, true=beyond.std_out,
                             false=missing.std_out, name="JumpTarget"))

        frame = Frame(tree, location=(12.0, -7.4), label="FindTheBracket")
        frame.add(pieces + [zone])
        return answer.std_out

    # ----------------------------------------------------------------
    @classmethod
    def simulate(cls, memory, steps=8192):
        """Run *memory* as its own program in python, as the node runs it.

        The reference is ``brainfuck/bff/bff.py``, the port of the authors'
        interpreter, and this is deliberately a second implementation of the
        same eleven lines rather than a call into it: what it is for is to say
        what the *node* should produce, and a check that shares its code with
        the thing it checks is no check at all.

        The rules, all of them:

        * the instruction is ``memory[counter]``; anything that is not one of
          the ten is a no-op, and there are a great many of those in a soup
        * the two heads wrap round the ring of memory, the counter does not -
          it runs off the end and the machine stops
        * ``+`` and ``-`` are byte arithmetic
        * a bracket finds its partner by scanning memory as it stands *now*,
          which is why a program that rewrites itself still runs; an unmatched
          one sends the counter off the end

        :param memory: the cells to start with, the tapes end to end.
        :param steps: how many instructions to run at most. The paper's own
            limit is 8192, and a replicator reaches it - it never halts.
        :return: ``(steps, memory, head0, head1, counter)`` - the number of
            instructions actually executed before the machine ran off the end,
            the memory it leaves behind, and where the two heads and the
            counter ended up.
        """
        memory = list(memory)
        size = len(memory)
        counter = head0 = head1 = 0
        done = 0
        while done < steps and 0 <= counter < size:
            code = memory[counter]
            onward = counter + 1
            if code == cls.RIGHT:
                head0 = (head0 + 1) % size
            elif code == cls.LEFT:
                head0 = (head0 - 1) % size
            elif code == cls.BRACE_RIGHT:
                head1 = (head1 + 1) % size
            elif code == cls.BRACE_LEFT:
                head1 = (head1 - 1) % size
            elif code == cls.PLUS:
                memory[head0] = (memory[head0] + 1) % 256
            elif code == cls.MINUS:
                memory[head0] = (memory[head0] - 1) % 256
            elif code == cls.DOT:
                memory[head1] = memory[head0]
            elif code == cls.COMMA:
                memory[head0] = memory[head1]
            elif code == cls.OPEN and memory[head0] == 0:
                onward = cls._partner(memory, counter, 1) + 1
            elif code == cls.CLOSE and memory[head0] != 0:
                onward = cls._partner(memory, counter, -1) + 1
            counter, done = onward, done + 1
        return done, memory, head0, head1, counter

    @classmethod
    def _partner(cls, memory, counter, step):
        """The bracket matching the one at *counter*, searching in *step*'s direction.

        The python of :meth:`_bracket_scan`, and the same walk: start one cell
        on, one deep, and count a bracket of the same kind as one deeper and
        its partner as one shallower until the count runs out.

        :return: the index of the partner, or one past the end of memory in
            the direction of the search when there is none - which the caller
            turns into a counter that halts the machine.
        """
        size = len(memory)
        at, depth = counter + step, 1
        while depth > 0 and 0 <= at < size:
            if memory[at] == cls.OPEN:
                depth += step
            elif memory[at] == cls.CLOSE:
                depth -= step
            at += step
        if depth != 0:
            return size if step > 0 else -2
        return at - step


class BrainFuckExtendedModifier(GeometryNodesModifier):
    """
    A whole brainfuck machine running inside geometry nodes: two tapes of cells
    whose values are incremented, decremented and copied about, two heads that
    walk along them, and a program that is consumed one instruction per
    ``step_duration``.

    This is the python translation of the ``SimpleBrainFuck`` graph in
    ``video_bff/tmp.xml``, completed and debugged, and then given the second
    head of the BFF paper. Every frame of that graph is built by a private
    method of its own.

    **The instructions.** Ten of them, and which head each one works on is the
    whole of what there is to know:

    ``>`` ``<``
        move head0, drawn below the tape as an arrow and marked by the square
        that runs along with it.
    ``{`` ``}``
        move head1, drawn above the tape. Nothing else touches it - it is a
        second head with no arithmetic of its own.
    ``+`` ``-`` ``[`` ``]``
        add to, take from, and loop on the cell under head0.
    ``.`` ``,``
        copy. ``.`` writes the cell under head0 into the cell under head1 and
        ``,`` writes it back the other way. Nothing is printed - what a
        program has to say it says by writing it onto the tape, which is what
        lets a program write another program.

    **The encoding.** ``code_table`` is the printable ascii range from 32 up,
    so a cell holds the code of the character the table shows against it and a
    cell holding 43 holds a ``+``. A cell holding one of the ten instructions
    draws it instead of its number - twice the size and in the colour that
    instruction has in the table above the tape - so that a tape carrying a
    program can be read as a program rather than as 64 numbers.

    **The state.** A simulation zone carries

    ``Geometry``
        the tape itself. This is the answer to "how do the cell values get
        incremented" - the values live in the integer point attribute
        ``Value`` of the tape geometry, and the zone hands that geometry from
        one frame to the next. ``+`` and ``-`` are a single ``Store Named
        Attribute`` whose *selection* is ``Index == PointerPosition``, so only
        the cell under head0 changes, and a second one selected on
        ``Index == MatePosition`` is the whole of ``.``; ``Sample Index``
        reads the two cells back out for the copies and for the increment
        itself.

        What the two tapes *start* with comes out of two csv files rather than
        out of the graph: an ``Import CSV`` node per tape in the control frame
        reads :attr:`TAPE_FILES` - ``data/replicator.csv`` for the upper tape
        and ``data/food.csv`` for the lower one - and the tape frame samples
        the resulting point cloud by cell index. One number per line, and a
        header line, because ``Import CSV`` spends the first line of the file
        on the name of the column, so the soup of the paper can be swapped for
        another without touching a single node.

    ``Counter``
        the program counter - an index into memory, because the program *is*
        memory. The machine reads the cell the counter points at and takes its
        value for an opcode; anything that is not one of the ten is a no-op.
        That is what lets a tape be a program, and what makes a program that
        copies a tape a program that copies a program.

    ``PointerPosition``, ``MatePosition``, ``Step``, ``StartTime``, ``Time``
        where the two heads are, the index of the current step, and the clock.
        Both heads start on cell 0 and both run round a *ring* of memory: the
        two tapes are laid end to end, cell 0 to 63 on the tape drawn above and
        64 to 127 on the one below, and a head walking left off cell 0 comes
        back on at cell 127. That is the whole reason a program on the first
        tape can write onto the second one - a ``{`` on the first step is
        enough to put a head at the far end - and it is what the paper's
        interpreter does with its ``head &= 127``. The counter is the one
        thing that does not wrap: it runs off the end and the machine stops.

    **The jumps.** Where a jump goes *is* searched for at run time, in the tape
    as it stands, by :meth:`_create_bracket_scan` - a repeat zone walking out
    from the bracket, one deeper for each bracket of its own kind and one
    shallower for its partner, until the count runs out. The one-headed machine
    matches its brackets in python when the graph is built and bakes the answer
    into a constant, which is cheaper and no use at all here: the program is
    the tape, so the very next ``.`` can move a bracket, and a soup of random
    bytes has unmatched brackets in it that no table can describe. An unmatched
    one sends the counter off the end of memory, where the machine halts -
    which is what the paper's interpreter does with it.


    ``Time`` accumulates ``Delta Time`` rather than reading the scene clock, so
    the machine is driven by the simulation itself. Nothing happens before
    ``start_time``; after that the step index is
    ``floor((Time - StartTime) / StepDuration)`` and an instruction is executed
    on every frame where that index goes up *and* the program has not run out.
    The xml carries a second state item ``OldStep`` for the comparison, but it
    is written from the same socket as ``Step`` and so always holds the same
    value; ``Step`` alone is the previous step index, and that is all the
    comparison needs.

    Because the state lives in a simulation zone, the machine only runs when
    blender steps the scene forward one frame at a time. Jumping straight to a
    frame shows whatever the zone last cached, and ``render_with_skips`` will
    treat frames as still unless some *object* is animated - see the scene
    ``BffScene.simple_brain_fuck`` for what that means in practice.

    The frames of the graph:

    ``ControlParameter``, ``Variables``
        the constants - among them the two csv files the tape is filled
        from - and the four numbers the machine starts with (``Counter``,
        ``PointerPosition``, ``MatePosition``, ``Step``).

    ``Tape``
        two ``Mesh Line`` nodes of ``TapeSize`` points each, joined into one
        run of memory and filled from the csv files - the tape as the machine
        starts it.

    ``RunProgram``
        the simulation zone: the clock, the instruction under the counter and
        the handover of every state item.

    ``Automaton``
        the instruction decoder. ``Char To Ascii`` turns the current
        instruction into its code, one ``Compare`` per opcode fires, and each
        comparison is ``AND``-ed with "a step has just begun" so that an
        instruction takes effect exactly once however many frames it is on
        screen for. It also works out where the counter goes next, which for
        eight of the ten instructions is simply one on.

    ``Cells``, ``CellValues``
        the tape as it looks: one square per cell, grey unless the cell holds
        the code of an instruction, and painted again where head0 is, with the
        value written on it as a number by a *for each element* zone - or,
        where that number is the ascii code of one of the ten instructions,
        the instruction itself, twice the size and in its colour, so that a
        tape carrying a program can be read as one.

    ``Heads``
        the two arrows and the square that marks the active cell. Head0 hangs
        below the cell it is on in the colour of ``<``, head1 stands above its
        own cell in the colour of ``{``, and the square stands around the cell
        head0 is on. All three ride along with the tape: they are joined in
        before the tilt and carry the ``Tape`` attribute of the cell they
        point at, so each lands on the line of its own tape.

    ``CodeTable``, ``TableFrame``
        the printable ascii characters and their codes in a row above the
        tape, drawn by a repeat zone, with a rectangle around it that is
        sized from the bounding box of what came out - so it fits whatever
        table is passed in.

    :param code_table: the characters the cell values are the codes of
    :param tape_size: cells per tape; there are two of them
    :param cell_size: width and height of a single cell
    :param step_duration: seconds one instruction is on screen. Must be at
        least two frames; below that the machine simply runs at one
        instruction per frame instead of skipping any.
    :param start_time: seconds before the first instruction runs
    :param tape_tilt: angle the tape is laid back by, so that a camera looking
        along +y sees the faces of the cells rather than their edge
    :param glyph_size: height of the number on a cell, as a fraction of
        ``CellSize``
    :param display_height: height of a read-out box
    :param colors: optional ``{node name: colour name}`` overriding
        :attr:`CELL_COLORS`, :attr:`PROGRAM_COLORS`, any instruction of
        :data:`INSTRUCTION_COLORS`, and the two entries ``GlyphColor`` and
        ``FrameColor``
    :param tape_files: one csv file per tape, resolved against ``DATA_DIR``;
        their contents are the tape, and so the program
    """

    ALPHABET = " !" + chr(
        34) + r"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"

    # the ten instructions, as characters: what to colour and what to draw
    # in place of a number. The machine itself knows them as codes - see
    # :attr:`BFFNode.COMMANDS`, which this is the same string as.
    COMMANDS = BFFNode.COMMANDS
    # how much bigger an instruction is drawn than the rest of the alphabet
    command_glyph_scale = 4
    # how much bigger a cell holding the ascii code of an instruction draws
    # that instruction than it would have drawn the number
    cell_command_scale = 2
    # Colour of a cell by what is in it. The name is the name of the
    # ``Input Material`` node in the control frame, so any of them can be
    # swapped or animated through
    # ``ibpy.get_geometry_node_from_modifier(mod, "PointerColor")``.
    #
    # The chain is applied in this order and each link overrides the previous
    # one, so the first entry is the fall-back and the last one wins.
    CELL_COLORS = (
        ("ZeroColor", "gray_4"),  # anything that is not an instruction
        ("ValueColor", "drawing"),  # holds the code of one of the ten
        ("PointerColor", "important"),  # the cell head0 is on, and its square
    )
    GLYPH_COLOR = "text"  # the numbers on the cells and all the text
    FRAME_COLOR = "gray_2"  # the boxes around the displays and the code table

    # Colour of an instruction in the program strip by what has become of it.
    # The instruction being executed is painted in ``PointerColor``, the same
    # colour as the head marker, so the two read as one thing.
    #
    # Applied in this order, each overriding the last: the fall-back is "has
    # not run yet", then "has run", then "has run but is inside a loop that is
    # still open, so it will run again".
    PROGRAM_COLORS = (
        ("ProgramColor", "text"),  # still to come
        ("DoneColor", "gray_2"),  # run, and not coming back
        ("WaitingColor", "example"),  # waiting for the next turn of its loop
    )

    # Colour of the instructions in the ascii table: the shared palette, so
    # that a tape drawn by another class and this table can be shown together
    # and an instruction is the same colour in both.
    OPCODE_COLORS = INSTRUCTION_COLORS

    # how the code table is laid out: one entry every ``table_spacing`` along
    # x, the letter ``table_line_gap`` below its number, and a frame around it
    # that is ``table_margin`` times the extent of the whole row
    table_spacing = 1
    table_line_gap = 0.7
    table_glyph_size = 0.5
    table_margin = 1.1
    frame_radius = 0.03

    # the program strip: one column per instruction, each column this much
    # wider than the letter that stands in it
    strip_glyph_size = 1.4
    # the box that runs along the strip marking the instruction about to be
    # executed - so many columns wide, and so much of the height of the display
    # it runs inside
    cursor_width = 1.3
    cursor_height = 0.7

    # the gap between the input display and the output display below it
    display_gap = 0.6

    # The two heads, in cells. An arrow is built about its middle: blender's
    # cone runs from z=0 to depth while its cylinder is centred on the origin,
    # so the stem is dropped by half its own length to meet the base of the
    # head rather than by half the arrow.
    arrow_length = 1.6
    arrow_width = 0.7
    arrow_gap = 0.35  # between the point of an arrow and the cell it marks
    # the square that marks the cell the counter is on - the instruction
    # being executed - and how thick its wire is, both in cells
    cursor_scale = 1.3
    cursor_weight = 0.06
    #: how much taller than wide the cursor is, and how far its middle sits
    #: above the middle of the cell - a square on the cell by default, but a
    #: subclass can make it a rectangle tall enough to take in the character
    #: standing on the cell as well. See :class:`BrainFuckHelloModifier`.
    cursor_tall = 1.0
    cursor_lift = 0.0

    #: where the counter starts, and the cell it stops on. ``None`` is the end
    #: of memory, which is the only stop a machine reading a soup can know; a
    #: machine running a program that was put there on purpose knows where the
    #: program begins and ends. See :class:`BrainFuckHelloModifier`.
    first_instruction = 0
    halt_at = None

    # The two tapes and where they get their initial content from: an Import
    # CSV node per tape in the control frame, named after the node it becomes
    # and reading the file of the same position in ``tape_files``. Tape 0 is
    # the one drawn on top - the self-replicator - and tape 1 the food below
    # it. One number per line, and a header line, since Import CSV spends the
    # first line of the file on the name of the column. The names are given
    # without the ".csv" - that and DATA_DIR are added when the path is built.
    TAPE_SOURCES = ("ReplicatorData", "FoodData")
    TAPE_FILES = ("replicator", "food")

    def __init__(self, code_table=None, table_width=30, tape_size=64, cell_size=0.25,
                 step_duration=0.5, start_time=3.0, tape_tilt=0.4607669,
                 glyph_size=0.6, display_height=2.0, colors=None, tape_files=None,
                 name="SimpleBrainFuck", **kwargs):
        # there is no ``program`` argument: the program is whatever the csv
        # files put on the tape, and the counter reads it from there
        self.code_table = self.ALPHABET if code_table is None else code_table
        self.command_table = self.COMMANDS
        self.table_width = table_width
        self.tape_size = tape_size
        self.cell_size = cell_size
        self.step_duration = step_duration
        self.start_time = start_time
        self.tape_tilt = tape_tilt
        self.glyph_size = glyph_size
        self.display_height = display_height
        # one csv file per tape, resolved against DATA_DIR
        self.tape_files = tuple(self.TAPE_FILES if tape_files is None else tape_files)
        # the column the values stand in - whatever the header line of the file
        # says, so that the Named Attribute nodes read what Import CSV wrote
        self.tape_columns = tuple(csv_column(os.path.join(DATA_DIR, file_name + ".csv"))
                                  for file_name in self.tape_files)
        overrides = colors or {}
        self.cell_colors = tuple((node_name, overrides.get(node_name, color))
                                 for node_name, color in self.CELL_COLORS)
        self.opcode_colors = tuple(
            (node_name, overrides.get(node_name, color), characters)
            for node_name, color, characters in self.OPCODE_COLORS)
        self.program_colors = tuple((node_name, overrides.get(node_name, color))
                                    for node_name, color in self.PROGRAM_COLORS)
        self.glyph_color = overrides.get("GlyphColor", self.GLYPH_COLOR)
        self.frame_color = overrides.get("FrameColor", self.FRAME_COLOR)
        self.kwargs = kwargs
        super().__init__(name=name, automatic_layout=False)

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        # Coord(tree, min=(-10, 0), max=(10, 20))

        control = self._create_control_frame(tree)
        variables = self._create_variables_frame(tree)
        tape = self._create_tape_frame(tree, control)
        run = self._create_run_program_frame(tree, control, variables, tape)

        cells = self._create_cells_frame(tree, control, run)
        table = self._create_code_table_frame(tree, control)

        # simulated = self._create_simulated_geometry_frame(tree, control, variables, run)

        out = self.group_outputs
        out.location = (38 * 200, -2 * 200)
        join = JoinGeometry(tree, location=(36, -4))
        for piece in [cells, table]:  #, simulated]:
            tree.links.new(piece, join.geometry_in)
        tree.links.new(join.geometry_out, out.inputs["Geometry"])

    # ----------------------------------------------------------------
    def _create_control_frame(self, tree):
        """``ControlParameter``: every constant of the machine.

        :return: ``{name: node}``, so that the frames downstream can pick the
            parameter they need by the name it carries in the editor.
        """
        x = -23.8
        control = {
            "TableWidth": InputInteger(tree, location=(x, -0.8), integer=self.table_width, hide=True),
            "TapeSize": InputInteger(tree, location=(x, 0), integer=self.tape_size,
                                     name="TapeSize", hide=True),
            "CellSize": InputValue(tree, location=(x, -0.8), value=self.cell_size,
                                   name="CellSize", hide=True),
            "StartTime": InputValue(tree, location=(x, -1.6), value=self.start_time,
                                    name="StartTime", hide=True),
            "StepDuration": InputValue(tree, location=(x, -2.4), value=self.step_duration,
                                       name="StepDuration", hide=True),
            "CodeTable": InputString(tree, location=(x, -3.2), string=self.code_table,
                                     name="CodeTable", hide=True),
            "CommandTable": InputString(tree, location=(x, -3.8), string=self.command_table, name="CommandTable",
                                        hide=True)
        }

        # one Input Material node per colour of a cell, plus the two that
        # everything else is drawn in
        palette = {}
        rows = ([(node_name, color) for node_name, color in self.cell_colors]
                + [(node_name, color) for node_name, color, _ in self.opcode_colors])

        for row, (node_name, color) in enumerate(rows):
            palette[node_name] = InputMaterial(tree, location=(x, -4.4 - 0.4 * row),
                                               material=color, name=node_name,
                                               **self.kwargs, hide=True)
        rest = list(self.program_colors) + [("GlyphColor", self.glyph_color),
                                            ("FrameColor", self.frame_color)]
        for offset, (node_name, color) in enumerate(rest):
            palette[node_name] = InputMaterial(
                tree, location=(x, -4.4 - 0.4 * (len(rows) + offset)),
                material=color, name=node_name, **self.kwargs)
        for source in palette.values():
            self.materials.append(source.node.material)
        control.update(palette)

        # The ten instruction colours travel as one bundle rather than as ten
        # links. Every frame that paints an instruction wants all ten - the
        # code table above the tape, the numbers on the cells - so what crossed
        # the graph as ten wires from here to each of them now crosses as one,
        # and the frame that receives it says what it is unpacking in a single
        # Separate Bundle. The Input Material nodes stay where they are: the
        # bundle gathers them rather than replacing them, so each colour is
        # still a node of its own to reach for by name.
        control["OpColors"] = CombineBundle(
            tree, location=(x + 1.6, -4.4), name="OPColorBundle",
            items=[(node_name, "MATERIAL", palette[node_name].std_out)
                   for node_name, _, _ in self.opcode_colors])

        # The read-outs below the tape and the code table above it are all
        # centred on the middle of the tape, which runs from x=0 to
        # x=TapeSize*CellSize. Everything the machine shows is therefore
        # stacked in one column and a camera looking along +y frames it whole.
        #

        # where the code table starts - its entries grow to the right from
        # here, so it is shifted left by half its own width - and how far the
        # head marker hangs below the tape
        table_start = - 0.5 * (self.table_width - 1) * self.table_spacing
        control["TablePosition"] = InputVector(tree, location=(x, -11.0),
                                               vector=Vector([table_start, 0, 10]),
                                               name="TablePosition")
        control["PointerOffset"] = InputVector(tree, location=(x, -11.8),
                                               vector=Vector([0, 0, -0.9 * self.cell_size]),
                                               name="PointerOffset")
        # how far the middle of an arrow sits from the middle of the cell it
        # marks: half a cell to clear the square, the gap, and half the arrow,
        # since an arrow is built about its own middle
        control["ArrowOffset"] = InputValue(
            tree, location=(x, -12.2), name="ArrowOffset",
            value=(0.5 + self.arrow_gap + 0.5 * self.arrow_length) * self.cell_size,
            hide=True)

        # what the two tapes start with. One row of the file per cell, read at
        # render time - the tape frame samples these point clouds by index, so
        # a different soup is a different file rather than a different graph.
        for row, (node_name, file_name) in enumerate(zip(self.TAPE_SOURCES, self.tape_files)):
            control[node_name] = ImportCSV(tree, location=(x, -12.6 - 1.2 * row),
                                           path=os.path.join(DATA_DIR, file_name + ".csv"),
                                           name=node_name, label=file_name)

        frame = Frame(tree, location=(-24, 0.6), label="ControlParameter")
        frame.add(list(control.values()))
        return control

    # ----------------------------------------------------------------
    def _create_variables_frame(self, tree):
        """``Variables``: the four numbers the machine starts with."""
        x = -15.8
        variables = {
            # No program and no jump table: the program is the tape, and where
            # a bracket jumps to is searched for in the tape as it stands - see
            # :meth:`_create_bracket_scan`. What is left is the four numbers the
            # machine starts with.
            "Pointer": InputInteger(tree, location=(x, -3.2), integer=0,
                                    name="PointerPosition"),
            # Both heads start on the first cell of the first tape, where the
            # paper starts them. Which tape a head is on is not a property of
            # the head: memory is one ring of TapeSize * 2 cells and the two
            # tapes are two halves of it, so a head walks from one onto the
            # other simply by carrying on.
            "Mate": InputInteger(tree, location=(x, -3.6), integer=0,
                                 name="MatePosition"),
            "Counter": InputInteger(tree, location=(x, -4.0),
                                    integer=self.first_instruction,
                                    name="ProgramCounter"),
            # -1, so that the first step (index 0) counts as an advance and the
            # first instruction is executed rather than skipped
            "Step": InputInteger(tree, location=(x, -4.8), integer=-1, name="Step"),
        }
        frame = Frame(tree, location=(-16, 0.6), label="Variables")
        frame.add(list(variables.values()))
        return variables

    # ----------------------------------------------------------------
    def _create_tape_frame(self, tree, control):
        """``Tape``: the cells as the machine starts them, filled from the csv files.

        Cell *n* of tape *i* starts with row *n* of ``tape_files[i]`` -
        ``replicator.csv`` on the tape above, ``food.csv`` on the one below.
        The file is read by an ``Import CSV`` node in the control frame, which
        turns it into a point cloud of one point per row, and a ``Sample
        Index`` picks the row whose number matches the index of the cell.

        :return: the geometry socket of the initial tape.
        """
        frame = Frame(tree, location=(-8.2, 1.4), label="Tape")
        tape_join = JoinGeometry(tree, location=(-2.6, -1.5), name="TapeJoin", hide=True)

        # create two frames that are connected by the turing machine
        ends = []
        for i in range(2):
            length = MathNode(tree, location=(-8, -3 * i), operation="MULTIPLY",
                              inputs0=control["TapeSize"].std_out,
                              inputs1=control["CellSize"].std_out, name="TapeLength")
            end = CombineXYZ(tree, location=(-7, -3 * i), x=length.std_out, name="TapeEnd")
            line = MeshLine(tree, location=(-6, 0.6 - 3 * i), mode="END_POINTS",
                            count=control["TapeSize"].std_out,
                            start_location=Vector([0, 0, 0]), end_location=end.std_out)
            # what the file holds for this cell. The column of the point cloud
            # is named after the header line of the csv file, and the index is
            # the index of the cell being written - a cell beyond the end of
            # the file gets a zero, so a short file simply leaves the rest of
            # the tape blank.
            column = NamedAttribute(tree, location=(-6, -0.6 - 3 * i), data_type="INT",
                                    name=self.tape_columns[i], label="CsvColumn")
            cell = Index(tree, location=(-6, -1.2 - 3 * i), name="CellIndex", hide=True)
            content = SampleIndex(tree, location=(-5.4, 0.6 - 3 * i), data_type="INT",
                                  domain="POINT", geometry=control[self.TAPE_SOURCES[i]].geometry_out,
                                  value=column.std_out, index=cell.std_out,
                                  label="ReadCell" + str(i))
            # the attribute has to exist from the first frame on, otherwise the
            # "Sample Index" in the automaton has nothing to read and the cells
            # have nothing to be coloured by.
            values = StoredNamedAttribute(tree, location=(-4.6, 0.6 - 3 * i), data_type="INT",
                                          domain="POINT", name="Value", value=content.std_out,
                                          label="LoadTape")

            tape_kind = StoredNamedAttribute(tree, location=(-3.6, 0.6 - 3 * i), data_type="INT",
                                             domain="POINT", name="Tape", value=i,
                                             label="TapeNumber")
            # Which cell of the whole memory this is: the two tapes are laid
            # end to end, so tape 1 starts at TapeSize. It is written down here
            # for two reasons. The point index only says it before the join,
            # and the drawn cells are instances by the time anything asks -
            # their point indices count vertices, not cells. So "the cell a
            # head is on" is picked out of the realized geometry by reading
            # this attribute back rather than by comparing Index.
            offset = [] if i == 0 else [
                IntegerMath(tree, location=(-3.4, -0.4 - 3 * i), operation="ADD",
                            inputs0=cell.std_out, inputs1=control["TapeSize"].std_out,
                            name="CellOnTape" + str(i), hide=True)]
            number = StoredNamedAttribute(tree, location=(-3.0, 0.6 - 3 * i), data_type="INT",
                                          domain="POINT", name="Cell",
                                          value=offset[0].std_out if offset else cell.std_out,
                                          label="CellNumber")
            create_geometry_line(tree, [line, values, tape_kind, number])
            ends.append(number)
            frame.add([length, end, line, column, cell, content, values, tape_kind,
                       number, tape_join] + offset)

        # Backwards, because blender puts the newest link into a multi-input
        # socket on top and Join Geometry concatenates top to bottom: linking
        # tape 1 first is what makes cell 0 to 63 the tape drawn above -
        # ``replicator.csv`` - and 64 to 127 the one below it, so that a head
        # walking off the end of the program lands on the food.
        for piece in reversed(ends):
            tree.links.new(piece.geometry_out, tape_join.geometry_in)

        return tape_join.geometry_out

    # ----------------------------------------------------------------
    def _create_run_program_frame(self, tree, control, variables, tape):
        """``RunProgram``: the simulation zone - the clock and the program counter.

        The counter indexes *memory*, not a program string: the machine reads
        its next instruction out of the cell the counter points at, which is
        the whole of what makes this the BFF machine of the paper rather than
        a brainfuck interpreter with a tape beside it.

        :return: ``{name: socket}`` of the state as it leaves the zone.
        """
        zone = Simulation(tree, location=(2, 5), node_width=9, geometry=tape)
        sim_in, sim_out = zone.simulation_input, zone.simulation_output
        for socket_type, socket_name, initial in (
                ("FLOAT", "StartTime", control["StartTime"].std_out),
                ("INT", "Step", variables["Step"].std_out),
                ("INT", "PointerPosition", variables["Pointer"].std_out),
                ("INT", "MatePosition", variables["Mate"].std_out),
                ("INT", "Counter", variables["Counter"].std_out),
                ("FLOAT", "Time", 0.0)):
            zone.add_socket(socket_type=socket_type, name=socket_name, value=initial)

        # --- the clock -------------------------------------------------
        # the zone's own Delta Time rather than the scene clock, so that the
        # machine keeps its own time and a state item is all that is needed
        time = MathNode(tree, location=(3.2, 6.4), operation="ADD",
                        inputs0=sim_in.outputs["Delta Time"],
                        inputs1=sim_in.outputs["Time"], name="Clock")
        since = MathNode(tree, location=(4.4, 6.4), operation="SUBTRACT",
                         inputs0=time.std_out, inputs1=sim_in.outputs["StartTime"],
                         name="SinceStart")
        scaled = MathNode(tree, location=(5.6, 6.4), operation="DIVIDE",
                          inputs0=since.std_out,
                          inputs1=control["StepDuration"].std_out, name="InSteps")
        # -1 while the machine is still waiting, so that the first real step
        # (index 0) is an increase and fires the first instruction
        waiting = MathNode(tree, location=(6.8, 6.4), operation="MAXIMUM",
                           inputs0=scaled.std_out, inputs1=-1.0, name="NotBeforeStart")
        # FLOOR, not TRUNC: truncation rounds towards zero, so the whole last
        # step_duration before StartTime would already come out as step 0 and
        # fire the first instruction early
        step = MathNode(tree, location=(8.0, 6.4), operation="FLOOR",
                        inputs0=waiting.std_out, name="StepIndex")
        # An instruction is executed on the one frame where the step index goes
        # up, never on the frames in between - otherwise a single "+" would
        # count once per rendered frame. Comparing against the *previous* index
        # rather than using the difference as a count also means that a
        # step_duration shorter than a frame degrades to one instruction per
        # frame instead of skipping instructions.
        advance = CompareNode(tree, location=(9.2, 6.4), operation="GREATER_THAN",
                              data_type="INT", inputs0=step.std_out,
                              inputs1=sim_in.outputs["Step"], name="Advance")

        # --- has the machine run off the end of memory? -------------------
        # The clock keeps going after the last instruction, so without this the
        # machine would go on "executing" past the end of memory. The counter
        # does not wrap the way the two heads do - the paper's interpreter
        # stops when it runs off either end, and halting leaves the finished
        # state up. How long memory is comes off the tape rather than out of
        # TapeSize, so that this says the same thing as the Domain Size inside
        # the machine node.
        size = DomainSize(tree, location=(3.2, 3.0), component="MESH",
                          geometry=sim_in.outputs["Geometry"], name="MemorySize")
        cells = size.outputs["Point Count"]
        # the end of memory, unless the class knows a nearer one
        last = cells
        if self.halt_at is not None:
            stop = InputInteger(tree, location=(3.4, 3.0), integer=self.halt_at,
                                name="HaltAt", hide=True)
            last = stop.std_out
        inside = CompareNode(tree, location=(4.4, 3.6), operation="LESS_THAN",
                             data_type="INT", inputs0=sim_in.outputs["Counter"],
                             inputs1=last, name="BeforeTheEnd", hide=True)
        started = CompareNode(tree, location=(4.4, 3.2), operation="GREATER_THAN",
                              data_type="INT", inputs0=sim_in.outputs["Counter"],
                              inputs1=-1, name="AfterTheStart", hide=True)
        running = BooleanMath(tree, location=(5.4, 3.4), operation="AND",
                              inputs0=inside.std_out, inputs1=started.std_out,
                              name="NotHalted")
        fire = BooleanMath(tree, location=(10.4, 6.4), operation="AND",
                           inputs0=advance.std_out, inputs1=running.std_out,
                           name="ExecuteNow")

        # --- the reroutes that carry the step into the machine -------------
        fire_in = Reroute(tree, location=(11.6, 4.2), ins=fire.std_out, name="Fire")
        head_in = Reroute(tree, location=(11.6, 3.8),
                          ins=sim_in.outputs["PointerPosition"], name="Head0")
        mate_in = Reroute(tree, location=(11.6, 3.6),
                          ins=sim_in.outputs["MatePosition"], name="Head1")
        step_in = Reroute(tree, location=(11.6, 3.4), ins=sim_in.outputs["Counter"],
                          name="Counter")

        # One instruction of the machine, in a node group of its own: what the
        # ten instructions do is the same whatever tape it is done to, and this
        # frame is only here to say when to do it and to hand the answer back
        # to the simulation zone as state. The instruction itself is not passed
        # in - the node reads it out of the cell the counter points at, the
        # tape being the program.
        machine = BFFNode(tree, location=(13, 4.4), name="BFFMachine")
        for socket, socket_name in ((sim_in.outputs["Geometry"], "Geometry"),
                                    (head_in.std_out, "Head0"),
                                    (mate_in.std_out, "Head1"),
                                    (step_in.std_out, "Counter"),
                                    (fire_in.std_out, "Fire")):
            tree.links.new(socket, machine.inputs[socket_name])
        pointer = machine.outputs["Head0"]
        mate = machine.outputs["Head1"]
        counter = machine.outputs["Counter"]
        tape_out = machine.geometry_out

        for socket, name in ((time.std_out, "Time"), (step.std_out, "Step"),
                             (sim_in.outputs["StartTime"], "StartTime"),
                             (counter, "Counter"),
                             (pointer, "PointerPosition"), (mate, "MatePosition")):
            tree.links.new(socket, sim_out.inputs[name])
        # replaces the pass-through that the Simulation wrapper puts in
        tree.links.new(tape_out, sim_out.inputs["Geometry"])

        frame = Frame(tree, location=(1.6, 7.4), label="RunProgram")
        frame.add([zone, time, since, scaled, waiting, step, advance, fire,
                   inside, started, running, size, machine,
                   fire_in, head_in, mate_in, step_in])
        return {name: sim_out.outputs[name] for name in
                ("Geometry", "Step", "PointerPosition", "MatePosition", "Counter")}

    # ----------------------------------------------------------------
    def _create_cells_frame(self, tree, control, run):
        """``Cells``: the tape as it looks, coloured by what is on it.

        A filled square is instanced onto every tape point and the instances
        are realized, so that the point attributes of the tape reach the faces.
        A chain of ``Set Material`` then paints them: the first link has no
        selection and is the fall-back, each later one overrides it where its
        selection holds. A cell is coloured for holding an *instruction* rather
        than for holding any value at all - most of a soup is data, and
        colouring every non-zero cell paints the whole tape - and the cell the
        counter is on, the instruction being executed, is coloured again on
        top of that.

        The selections read the ``Cell`` attribute written in
        :meth:`_create_tape_frame` rather than ``Index``: these are realized
        instances, so an index here counts the vertices of the squares and not
        the cells.

        Everything that rides along with the tape - the numbers, the two
        arrows and the square around the active cell - is joined in before the
        tilt, so that one Transform Geometry lays the whole picture back and
        one Set Position drops each piece onto the line of its own tape.

        :return: the geometry socket of the finished tape.
        """
        tape = run["Geometry"]
        quad = Quadrilateral(tree, location=(26, 2), mode="RECTANGLE",
                             width=control["CellSize"].std_out,
                             height=control["CellSize"].std_out)
        fill = FillCurve(tree, location=(27, 2), mode="N-gons")
        create_geometry_line(tree, [quad, fill])
        instances = InstanceOnPoints(tree, location=(28, 2.6), points=tape,
                                     instance=fill.geometry_out)
        realize = RealizeInstances(tree, location=(29, 2.6))

        value = NamedAttribute(tree, location=(28, 1.2), data_type="INT", name="Value")
        here = NamedAttribute(tree, location=(28, 0.6), data_type="INT", name="Cell")
        holds = make_function(tree, name="CellHoldsACommand",
                              functions={"IsCommand": self._holds_code_of(self.command_table)},
                              inputs=["value"], outputs=["IsCommand"],
                              integers=["value"], booleans=["IsCommand"],
                              hide=True, location=(29, 1.2))
        tree.links.new(value.std_out, holds.inputs["value"])
        under = CompareNode(tree, location=(29, 0.6), operation="EQUAL",
                            data_type="INT", inputs0=here.std_out,
                            inputs1=run["Counter"], name="CellBeingExecuted",
                            hide=True)
        selections = (None, holds.outputs["IsCommand"], under.std_out)

        painters = [SetMaterial(tree, location=(30 + column, 2.6), selection=selection,
                                material=control[node_name].std_out,
                                name="Paint" + node_name)
                    for column, ((node_name, _), selection)
                    in enumerate(zip(self.cell_colors, selections))]
        create_geometry_line(tree, [instances, realize] + painters)

        numbers = self._create_cell_values(tree, control, run)
        heads = self._create_arrows_frame(tree, control, run)
        joined = JoinGeometry(tree, location=(34, 2.6))
        tree.links.new(painters[-1].geometry_out, joined.geometry_in)
        tree.links.new(numbers, joined.geometry_in)
        tree.links.new(heads, joined.geometry_in)
        # the tape lies in the x-y plane, which a camera looking along +y sees
        # edge-on. Laying it back brings the faces of the cells into view; the
        # numbers are pre-turned by the complement of this angle in
        # _create_cell_values, so that they come out upright.

        # lower position of second tape with the attribute

        attr_tape = NamedAttribute(tree, location=(31, 1.6), data_type="INT", name="Tape")

        tape_shift = make_function(tree, name="TapeShift",
                                   functions={
                                       "translation": ["cell_width,cell_number,*,-2,/", "0", "0"],
                                       "offset": ["0", "0", "-4,tape,*"]
                                   }, inputs=["cell_width", "cell_number", "tape"], outputs=["translation", "offset"],
                                   scalars=["cell_width", "cell_number"], integers=["tape"],
                                   vectors=["translation", "offset"],
                                   hide=True, location=(34, 1.6))
        tree.links.new(control["CellSize"].std_out, tape_shift.inputs["cell_width"])
        tree.links.new(control["TapeSize"].std_out, tape_shift.inputs["cell_number"])
        tree.links.new(attr_tape.std_out, tape_shift.inputs["tape"])

        tilt = TransformGeometry(tree, location=(35, 2.6), translation=tape_shift.outputs["translation"],
                                 rotation=[self.tape_tilt, 0, 0], name="LayTapeBack")

        set_position = SetPosition(tree, location=(36, 2), offset=tape_shift.outputs["offset"])

        create_geometry_line(tree, [joined, tilt, set_position])

        frame = Frame(tree, location=(25.6, 3.4), label="Cells")
        frame.add([quad, fill, instances, realize, value, here, holds, under,
                   joined, tilt, set_position, attr_tape, tape_shift] + painters)
        return set_position.geometry_out

    # ----------------------------------------------------------------
    def _create_arrows_frame(self, tree, control, run):
        """``Heads``: the two arrows, and the square around the active cell.

        ``Head0`` hangs below the cell it is on and ``Head1`` stands above it,
        each in the colour of the instructions that move it - ``<`` for the
        one moved by ``<`` and ``>``, ``{`` for the one moved by ``{`` and
        ``}``. ``Cursor`` is a wire square standing around the cell the
        *counter* is on - the instruction the machine is executing this step,
        which is a cell of the tape like any other, now that the tape is the
        program.

        Where each of them goes is read off the tape with ``Sample Index``
        rather than worked out again from ``TapeSize`` and ``CellSize``, so a
        head cannot drift away from the cells if the spacing of the Mesh Line
        is ever changed. The ``Tape`` attribute of that same cell is written
        onto the geometry, which is what makes the Set Position of
        :meth:`_create_cells_frame` drop a head onto the line of the tape it
        is currently on.

        :return: the geometry socket of the two arrows and the square, in the
            space of the tape - the tilt has not been applied yet.
        """
        at = Position(tree, location=(26, -8))
        on = NamedAttribute(tree, location=(26, -8.6), data_type="INT", name="Tape")
        pieces, marks = [], []
        for row, (label, index, colour, sign) in enumerate((
                ("Head0", run["PointerPosition"], "LessColor", -1.0),
                ("Head1", run["MatePosition"], "CurlyBraceOpenColor", 1.0))):
            y = -9 - 4 * row
            spot = SampleIndex(tree, location=(27, y), data_type="FLOAT_VECTOR",
                               domain="POINT", geometry=run["Geometry"],
                               value=at.std_out, index=index, name="CellOf" + label)
            line = SampleIndex(tree, location=(27, y - 0.6), data_type="INT",
                               domain="POINT", geometry=run["Geometry"],
                               value=on.std_out, index=index, name="TapeOf" + label)
            along = SeparateXYZ(tree, location=(28, y), vector=spot.std_out)
            # head0 hangs below its cell and head1 stands above it
            drop = MathNode(tree, location=(28, y - 1.2), operation="MULTIPLY",
                            inputs0=control["ArrowOffset"].std_out, inputs1=sign,
                            name="Offset" + label)
            where = CombineXYZ(tree, location=(30, y), x=along.x, z=drop.std_out,
                               name="Place" + label)
            tip = ConeMesh(tree, location=(27, y - 1.8), vertices=32, radius_top=0,
                           radius_bottom=0.5 * self.arrow_width * self.cell_size,
                           depth=0.5 * self.arrow_length * self.cell_size)
            stem = CylinderMesh(tree, location=(27, y - 2.4), vertices=32,
                                radius=0.25 * self.arrow_width * self.cell_size,
                                depth=0.5 * self.arrow_length * self.cell_size)
            below = TransformGeometry(tree, location=(28, y - 2.4),
                                      translation=[0, 0, -0.25 * self.arrow_length
                                                   * self.cell_size],
                                      name="Stem" + label)
            create_geometry_line(tree, [stem, below])
            body = JoinGeometry(tree, location=(29, y - 1.8))
            for piece in (tip.geometry_out, below.geometry_out):
                tree.links.new(piece, body.geometry_in)
            # the cone is born pointing along +z, which is right for the arrow
            # that points up at its cell from below; the other one is turned over
            turned = TransformGeometry(tree, location=(30, y - 1.8),
                                       rotation=[0, 0, 0] if sign < 0 else [pi, 0, 0],
                                       name="Turn" + label)
            put = TransformGeometry(tree, location=(31, y), translation=where.std_out,
                                    name="Put" + label)
            # one item out of the bundle rather than all ten: an arrow is
            # painted in the colour of the instructions that move it, and
            # nothing here cares about the other nine
            paint = GetBundleItem(tree, location=(31.4, y - 0.6),
                                  bundle=control["OpColors"].std_out, path=colour,
                                  socket_type="MATERIAL", name="ColorOf" + label,
                                  hide=True)
            painted = SetMaterial(tree, location=(32, y), material=paint.std_out,
                                  name="Paint" + label)
            # which tape the head is on, so that the Set Position downstream
            # moves the arrow to the same line as the cell it points at
            rides = StoredNamedAttribute(tree, location=(33, y), data_type="INT",
                                         domain="POINT", name="Tape",
                                         value=line.std_out, label="TapeOf" + label)
            create_geometry_line(tree, [body, turned, put, painted, rides])
            pieces += [spot, line, along, drop, where, tip, stem, below, body,
                       turned, put, paint, painted, rides]
            marks.append(rides)

        # --- the square around the instruction being executed ------------
        # Upright, turned by the complement of the tilt like the numbers are,
        # so that it comes out square to the camera. Lying in the plane of the
        # tape would be the obvious thing and is the wrong one: that plane is
        # laid back nearly flat, so a square drawn in it is seen almost
        # edge-on and disappears into the cell it is meant to mark.
        y = -17
        spot = SampleIndex(tree, location=(27, y), data_type="FLOAT_VECTOR",
                           domain="POINT", geometry=run["Geometry"], value=at.std_out,
                           index=run["Counter"], name="CellOfCursor")
        line = SampleIndex(tree, location=(27, y - 0.6), data_type="INT", domain="POINT",
                           geometry=run["Geometry"], value=on.std_out,
                           index=run["Counter"], name="TapeOfCursor")
        side = MathNode(tree, location=(27, y - 1.2), operation="MULTIPLY",
                        inputs0=control["CellSize"].std_out, inputs1=self.cursor_scale,
                        name="CursorSize")
        tall = MathNode(tree, location=(27.4, y - 1.8), operation="MULTIPLY",
                        inputs0=side.std_out, inputs1=self.cursor_tall,
                        name="CursorHeight")
        box = Quadrilateral(tree, location=(28, y - 1.2), mode="RECTANGLE",
                            width=side.std_out, height=tall.std_out)
        # a bare curve renders as a hair thin enough to disappear
        wire = CurveWireFrame(tree, location=(29, y - 1.2),
                              radius=self.cursor_weight * self.cell_size,
                              resolution=6, geometry=box.geometry_out)
        # the character stands above the middle of the cell, so a cursor that
        # takes it in has to rise with it
        lifted = VectorMath(tree, location=(29.4, y - 0.6), operation="ADD",
                            inputs0=spot.std_out,
                            inputs1=Vector([0, 0, self.cursor_lift]),
                            name="CursorPlace", hide=True)
        place = TransformGeometry(tree, location=(30, y),
                                  translation=lifted.std_out,
                                  rotation=[pi / 2 - self.tape_tilt, 0, 0],
                                  name="PlaceCursor")
        painted = SetMaterial(tree, location=(31, y),
                              material=control["PointerColor"].std_out,
                              name="PaintCursor")
        rides = StoredNamedAttribute(tree, location=(32, y), data_type="INT",
                                     domain="POINT", name="Tape", value=line.std_out,
                                     label="TapeOfCursor")
        create_geometry_line(tree, [place, painted, rides], ins=wire.geometry_out)
        pieces += [spot, line, side, tall, box, wire, lifted, place, painted, rides]

        joined = JoinGeometry(tree, location=(34, -12))
        for mark in marks + [rides]:
            tree.links.new(mark.geometry_out, joined.geometry_in)

        frame = Frame(tree, location=(25.6, -7.4), label="Heads")
        frame.add(pieces + [at, on, joined])
        return joined.geometry_out

    # ----------------------------------------------------------------
    @staticmethod
    def _holds_code_of(characters):
        """RPN for "the cell value is the ascii code of one of *characters*".

        The counterpart of the ``'<',letter,in`` of the code table, for a test
        on a *number* rather than on a letter: one ``Compare`` per character,
        or-ed together, so that one formula serves a single character and a
        whole set of them alike. The variable is called ``value`` and has to
        be an input of the function the result goes into.

        :param characters: the characters a colour stands for.
        :return: the RPN string.
        """
        test = "value,%d,=" % ord(characters[0])
        for character in characters[1:]:
            test += ",value,%d,=,or" % ord(character)
        return test

    # ----------------------------------------------------------------
    def _create_cell_values(self, tree, control, run):
        """``CellValues``: what every cell holds, written on it.

        A cell holding the ascii code of one of the ten instructions shows the
        instruction itself instead of the number - in the colour that
        instruction has in the code table above the tape, and
        :attr:`cell_command_scale` times as big. That is what lets a tape of
        numbers be read as a program, which is the point of a machine whose
        tape *is* its program: 43 on a cell is a number, ``+`` on a cell is
        code.

        The selector is the ``ColorSelector`` of
        :meth:`_create_code_table_frame` with its test swapped. There the
        entry of the table is a letter and ``Find in String`` asks whether it
        occurs among the characters a colour stands for; here the cell holds a
        *number*, so ``Compare`` asks whether it equals ``ord`` of any of
        them. Either way the result is one boolean per colour, driving the
        same chain of ``Set Material`` in which a later link overrides an
        earlier one wherever its selection holds.

        The character to draw comes out of ``CodeTable`` rather than out of a
        table of its own: that string is the printable range from 32 up, so
        the character of code *v* stands at position ``v - 32``.

        :return: the geometry socket of the numbers.
        """
        tape = run["Geometry"]
        value = NamedAttribute(tree, location=(26, -2), data_type="INT", name="Value")
        which = NamedAttribute(tree, location=(26, -3.2), data_type="INT", name="Cell")
        position = Position(tree, location=(25, -2.6))
        shift = VectorMath(tree, location=(26, -2.6), hide=True, operation="ADD", inputs0=position.std_out,
                           inputs1=Vector([0, 0, 0.25]))
        zone = ForEachZone(tree, location=(27, -1.4), domain="POINT", node_width=6,
                           geometry=tape)
        zone.add_socket(socket_type="INT", name="Value", value=value.std_out,
                        for_input=True)
        zone.add_socket(socket_type="VECTOR", name="Location", value=shift.std_out,
                        for_input=True)
        # which cell this is, carried in rather than read inside: an Index in
        # the body of the zone is the index of the element being drawn
        zone.add_socket(socket_type="INT", name="Cell", value=which.std_out,
                        for_input=True)
        held = zone.foreach_input.outputs["Value"]

        digits = ValueToString(tree, location=(28, -0.8), data_type="INT",
                               value=held, name="CellValue")
        rank = IntegerMath(tree, location=(28, -1.4), operation="SUBTRACT",
                           inputs0=held, inputs1=32, name="AsciiRank", hide=True)
        letter = SliceString(tree, location=(28.7, -1.4), string=control["CodeTable"].std_out,
                             position=rank.std_out, length=1, name="CellCommand")

        # one boolean per colour of :attr:`OPCODE_COLORS`, plus the one that
        # says whether the cell holds an instruction at all. All of them are
        # built from opcode_colors and command_table in a single call, so that
        # a colour standing for a pair of characters cannot drift out of step
        # with the list of selections below.
        socket_labels = [node_name for node_name, _, _ in self.opcode_colors]
        selectors = {node_name: self._holds_code_of(characters)
                     for node_name, _, characters in self.opcode_colors}
        selectors["IsCommand"] = self._holds_code_of(self.command_table)
        color_selection = make_function(tree, name="CellColorSelector",
                                        functions=selectors,
                                        inputs=["value"],
                                        outputs=socket_labels + ["IsCommand"],
                                        integers=["value"],
                                        booleans=socket_labels + ["IsCommand"],
                                        hide=True, location=(28.7, -3.2))
        tree.links.new(held, color_selection.inputs["value"])
        is_command = color_selection.outputs["IsCommand"]

        glyph = self._cell_glyph(tree, control, held, digits.std_out,
                                 letter.std_out, is_command,
                                 cell=zone.foreach_input.outputs["Cell"],
                                 counter=run["Counter"], location=(29.4, -0.8))

        size = MathNode(tree, location=(28, -2.4), operation="MULTIPLY",
                        inputs0=control["CellSize"].std_out, inputs1=self.glyph_size,
                        name="NumberSize")
        bigger = MathNode(tree, location=(28.7, -2.4), operation="MULTIPLY",
                          inputs0=size.std_out, inputs1=self.cell_command_scale,
                          name="CommandSize")
        glyph_size = Switch(tree, location=(29.4, -2.4), input_type="FLOAT",
                            switch=is_command, true=bigger.std_out, false=size.std_out,
                            name="GlyphSize")

        curves = StringToCurves(tree, location=(30, -1.4), string=glyph,
                                size=glyph_size.std_out, align_x="CENTER",
                                align_y="BOTTOM")
        realize = RealizeInstances(tree, location=(31, -1.4))
        fill = FillCurve(tree, location=(32, -1.4), mode="N-gons")
        # the fall-back paints every glyph and has no selection; the ten links
        # after it each override it on the cells holding their instruction
        opcolors = SeparateBundle(tree, location=(32.4, -3.2),
                                  bundle=control["OpColors"].std_out,
                                  items=control["OpColors"].items,
                                  name="OPColors")
        painters = ([SetMaterial(tree, location=(33, -1.4),
                                 material=control["GlyphColor"].std_out,
                                 name="PaintNumber")]
                    + [SetMaterial(tree, location=(33, -1.8 - 0.35 * row),
                                   selection=color_selection.outputs[node_name],
                                   material=opcolors.out(node_name),
                                   name="PaintCell" + node_name, hide=True)
                       for row, node_name in enumerate(socket_labels)])
        # the whole tape is laid back by tape_tilt further downstream, so a
        # number turned by the complement of that angle ends up standing
        # upright on a cell that is itself leaning away from the camera
        placed = TransformGeometry(tree, location=(34.4, -1.4),
                                   translation=zone.foreach_input.outputs["Location"],
                                   rotation=[pi / 2 - self.tape_tilt, 0, 0],
                                   name="PlaceNumber")
        zone.create_geometry_line([realize, fill] + painters + [placed],
                                  ins=curves.geometry_out)

        frame = Frame(tree, location=(25.6, -0.6), label="CellValues")
        frame.add([value, position, zone, digits, rank, letter, color_selection,
                   glyph, size, bigger, glyph_size, curves, realize, fill,
                   placed, opcolors] + painters)
        return zone.geometry_out

    # ----------------------------------------------------------------
    def _cell_glyph(self, tree, control, held, digits, letter, is_command,
                    cell=None, counter=None, location=(0, 0)):
        """What a cell draws: the instruction it holds, or else its number.

        Everything a cell can hold is a byte, and a byte is only an
        instruction if it is the code of one - so the number is the honest
        thing to draw for the rest. :class:`BrainFuckHelloModifier` puts a
        third case between the two.

        :param held: the value of the cell, an INT socket
        :param digits: that value written out, a STRING socket
        :param letter: the instruction it stands for, likewise
        :param is_command: whether it stands for one at all
        :param cell: which cell of memory this is, an INT socket
        :param counter: where the program counter stands, likewise - the two
            of them are what :class:`BrainFuckHelloModifier` needs to know
            that the machine has halted with an answer on these cells
        :return: the STRING socket of what to draw.
        """
        return Switch(tree, location=location, input_type="STRING",
                      switch=is_command, true=letter, false=digits,
                      name="CommandOrNumber").std_out

    # ----------------------------------------------------------------
    def _create_code_table_frame(self, tree, control):
        """``CodeTable``: ascii table from 32 to 126, framed.

        A repeat zone walks the table one character at a time - again because
        ``Slice String`` needs a single index - and joins the letter and its
        number into the geometry it carries. The frame around the result is a
        rectangle sized from the bounding box of what came out, so it fits
        whatever alphabet is passed in.

        :return: the geometry socket of the table.
        """
        table = control["CodeTable"]
        table_width = control["TableWidth"]
        size = StringLength(tree, location=(-14.4, 16.6), string=table.std_out,
                            name="TableLength")
        zone = RepeatZone(tree, location=(-13, 16), node_width=11,
                          iterations=size.std_out)

        origin = SeparateXYZ(tree, location=(-12, 18), hide=True,
                             vector=control["TablePosition"].std_out)
        column = make_function(tree, name="Column",
                               functions={
                                   "col": "iteration,width,%,spacing,*"
                               }, inputs=["iteration", "width", "spacing"], outputs=["col"],
                               scalars=["iteration", "width", "col", "spacing"], vectors=[], location=(-12, 15.4))
        tree.links.new(zone.iteration, column.inputs["iteration"])
        tree.links.new(table_width.std_out, column.inputs["width"])
        column.inputs["spacing"].default_value = self.table_spacing

        across = MathNode(tree, location=(-11, 17.4), operation="ADD",
                          inputs0=origin.x, inputs1=column.outputs["col"], name="AtColumn", hide=True)
        # the number sits on the line of TablePosition, the letter one line below
        row = make_function(tree, name="LetterLine",
                            functions={
                                "numberLine": "z,iteration,width,/,floor,lineSep,2,*,bandGap,+,*,-",
                                "letterLine": "z,lineSep,-,iteration,width,/,floor,lineSep,2,*,bandGap,+,*,-"
                            }, inputs=["iteration", "width", "z", "lineSep", "bandGap"],
                            outputs=["numberLine", "letterLine"],
                            scalars=["iteration", "lineSep", "bandGap", "numberLine", "letterLine", "z", "width"],
                            location=(-11, 15.4), hide=True)
        tree.links.new(table_width.std_out, row.inputs["width"])
        tree.links.new(origin.z, row.inputs["z"])
        tree.links.new(zone.iteration, row.inputs["iteration"])
        row.inputs["lineSep"].default_value = 0.7
        row.inputs["bandGap"].default_value = 0.3
        number_at = CombineXYZ(tree, location=(-9.5, 17.4), x=across.std_out,
                               y=origin.y, z=row.outputs["numberLine"],
                               name="NumberPosition", hide=True)

        letter_at = CombineXYZ(tree, location=(-9.5, 14.0), x=across.std_out,
                               y=origin.y, z=row.outputs["letterLine"],
                               name="LetterPosition", hide=True)

        letter = SliceString(tree, location=(-12, 14.6), string=table.std_out,
                             position=zone.iteration, length=1, name="Letter")

        # An instruction is drawn bigger than the rest of the alphabet. "in"
        # counts the letter in the command string - 1 for an instruction and 0
        # for anything else, since neither string repeats a character - and
        # "iff" picks a size from that.
        #
        # Operand order, which is the thing to get right: a custom op's
        # "inputs" tuple is (left, right, extra) while RPN pops right first,
        # left second and extra third. So a binary op reads left to right,
        # "haystack,needle,in", and the ternary reads
        # "condition,if_true,if_false,iff" - the same order as the "ifv" of
        # objects/hat_tile.py.
        custom_ops = {
            "in": {
                "type": FindInString,
                "inputs": ("String", "Search"),
                "output": "Count",
                "label": "in",
            },
            "iff": {
                "type": Switch,
                "class_kwargs": {"input_type": "FLOAT"},
                "inputs": ("True", "False", "Switch"),
                "output": "Output",
                "label": "iff",
            },

            "ifs": {
                "type": CompareNode,
                "class_kwargs": {"operation": "EQUAL", "data_type": "STRING"},
                "inputs": ("A", "B"),
                "output": "Result",
                "label": "StringEqual"
            }
        }

        letter_size = make_function(
            tree, name="LetterSize", custom_ops=custom_ops,
            functions={
                # Count is an integer, and the Switch wants a boolean; ",0,>"
                # says so rather than leaning on the implicit conversion
                "size": "commands,letter,in,0,>,plain,command_size,*,plain,iff",
            },
            inputs=["commands", "letter", "plain", "command_size"],
            outputs=["size"],
            scalars=["plain", "command_size", "size"],
            strings=["commands", "letter"],
            location=(-11, 14.0))
        letter_size.inputs["plain"].default_value = self.table_glyph_size / 2
        letter_size.inputs["command_size"].default_value = self.command_glyph_scale
        tree.links.new(letter.std_out, letter_size.inputs["letter"])
        tree.links.new(control["CommandTable"].std_out, letter_size.inputs["commands"])

        letter_curves = StringToCurves(tree, location=(-11, 13), string=letter.std_out,
                                       size=letter_size.outputs["size"], align_x="CENTER",
                                       align_y="MIDDLE", name="NumberToCurve", hide=True)
        # the table is read as "A is 1", so the label is the 1-based index
        rank = IntegerMath(tree, location=(-12, 17), operation="ADD",
                           inputs0=zone.iteration, inputs1=32, name="Rank")
        number = ValueToString(tree, location=(-11, 17), data_type="INT",
                               value=rank.std_out, name="RankLabel")
        number_curves = StringToCurves(tree, location=(-9.5, 13), string=number.std_out,
                                       size=self.table_glyph_size, align_x="CENTER",
                                       align_y="MIDDLE", hide=True, name="NumberToCurve")

        entries, ends = [], []
        # "y", not "row": "row" is the LetterLine node built above, and a
        # loop variable of that name leaves it holding 14.6 by the time the
        # frame is assembled - which parents a float and drops the node
        for curves, position, y, label in ((number_curves, number_at, 17.4, "Number"),
                                           (letter_curves, letter_at, 14.6, "Letter")):
            # String to Curves hands out instances of outlines; realizing and
            # filling them turns them into the solid letter that is drawn
            realize = RealizeInstances(tree, location=(-8, y))
            fill = FillCurve(tree, location=(-7, y), mode="N-gons")
            # the entry is one piece of geometry, not a field, so it can be
            # moved with Transform Geometry - Set Position would need it to be
            # an instance first and would then have to be realized again
            place = TransformGeometry(tree, location=(-6, y),
                                      translation=position.std_out,
                                      rotation=[pi / 2, 0, 0], name="Place" + label)
            create_geometry_line(tree, [realize, fill, place], ins=curves.geometry_out)
            entries += [realize, fill, place]
            ends.append(place)

        pair = JoinGeometry(tree, location=(-5, 16))
        for end in ends:
            tree.links.new(end.geometry_out, pair.geometry_in)

        # One boolean per colour of :attr:`OPCODE_COLORS`, saying whether this
        # entry of the table is one of the characters that colour stands for -
        # which is what the chain of Set Material below selects on. The labels
        # are built from opcode_colors in the same comprehension the selector
        # is, so that they cannot drift apart: two of its entries cover a
        # *pair* of characters, and a hand-written list of ten labels pairs up
        # with the eight colours wrongly and silently.
        socket_labels = [node_name for node_name, _, _ in self.opcode_colors]
        color_selection = instruction_selector(tree, letter.std_out,
                                               self.opcode_colors,
                                               location=(-5, 14))

        selections = [color_selection.outputs[label] for label in socket_labels]

        opcolors = SeparateBundle(tree, location=(-3.4, 15.4),
                                  bundle=control["OpColors"].std_out,
                                  items=control["OpColors"].items, name="OPColors")
        painters = ([SetMaterial(tree, location=[-2, 16.5], material=control["ZeroColor"].std_out, hide=True,
                                 name="PaintDefault")] +
                    [SetMaterial(tree, location=(-2, 16 - 0.5 * row), selection=selection,
                                 material=opcolors.out(node_name),
                                 name="Paint" + node_name, hide=True)
                     for row, ((node_name, _, _), selection)
                     in enumerate(zip(self.opcode_colors, selections))])

        create_geometry_line(tree, painters)

        grown = JoinGeometry(tree, location=(-1, 16))

        create_geometry_line(tree, [pair] + painters + [grown])
        zone.create_geometry_line([grown])

        box_x = 1
        box = self._create_table_frame(tree, control, zone.geometry_out, location=(box_x, 11))
        joined = JoinGeometry(tree, location=(box_x + 9, 16))
        tree.links.new(zone.geometry_out, joined.geometry_in)
        tree.links.new(box, joined.geometry_in)

        frame = Frame(tree, location=(-14.6, 18.4), label="CodeTable")
        frame.add([size, zone, origin, column, across, number_at, row, letter_at,
                   letter, letter_size, letter_curves, rank, number, number_curves,
                   pair, grown, joined, opcolors] + entries)
        return joined.geometry_out

    # ----------------------------------------------------------------
    def _create_table_frame(self, tree, control, table, location=(0, 0)):
        """The rectangle around the code table, sized from what it contains.

        :return: the geometry socket of the rectangle.
        """
        x, y = location
        y -= 11
        x += 3
        bounds = BoundingBox(tree, location=(x - 3, y + 14.6), geometry=table)
        extent = VectorMath(tree, location=(x - 2, y + 15.2), operation="SUBTRACT",
                            inputs0=bounds.max_out, inputs1=bounds.min_out,
                            name="TableExtent")
        margin = VectorMath(tree, location=(x - 1, y + 15.2), operation="SCALE",
                            inputs0=extent.std_out, float_input=self.table_margin,
                            name="WithMargin")
        sides = SeparateXYZ(tree, location=(x + 0, y + 15.2), vector=margin.std_out)
        middle = VectorMath(tree, location=(x - 2, y + 13.8), operation="ADD",
                            inputs0=bounds.min_out, inputs1=bounds.max_out,
                            name="TableCorners")
        centre = VectorMath(tree, location=(x - 1, y + 13.8), operation="SCALE",
                            inputs0=middle.std_out, float_input=0.5, name="TableCentre")
        # the table stands in the x-z plane, so its width and height are the x
        # and z of the bounding box, while the rectangle is born in x-y
        box = Quadrilateral(tree, location=(x + 1, y + 14.6), mode="RECTANGLE",
                            width=sides.x, height=sides.z)
        # a bare curve renders as a hair thin enough to disappear, so the
        # rectangle is given a body before it is drawn
        wire = CurveWireFrame(tree, location=(x + 2, y + 14.6), radius=self.frame_radius,
                              resolution=4, geometry=box.geometry_out)
        place = TransformGeometry(tree, location=(x + 3, y + 14.6), translation=centre.std_out,
                                  rotation=[pi / 2, 0, 0], name="PlaceTableFrame")
        painted = SetMaterial(tree, location=(x + 4, y + 14.6),
                              material=control["FrameColor"].std_out,
                              name="PaintTableFrame")
        create_geometry_line(tree, [wire, place, painted])

        frame = Frame(tree, location=(x - 4.4, y + 15.8), label="TableFrame")
        frame.add([bounds, extent, margin, sides, middle, centre, box,
                   wire, place, painted])
        return painted.geometry_out

    # ----------------------------------------------------------------
    def _create_cursor_frame(self, tree, control, counter, first, spacing, origin):
        """``CurrentDisplay``: the box that runs along the program strip.

        The same framed rectangle the read-outs are drawn with, sized to a
        single column of :meth:`_create_program_strip` and put where the
        counter points instead of somewhere fixed. It is painted in
        ``PointerColor``, the colour of the marker under the tape, so that the
        two heads - the one on the program and the one on the data - read as
        the same thing.

        :param counter: the program counter
        :param first: x of column 0 of the strip
        :param spacing: width of one column
        :return: the geometry socket of the box.
        """
        along = MathNode(tree, location=(21, -27.4), operation="MULTIPLY",
                         inputs0=counter, inputs1=spacing.std_out,
                         name="CursorOffset")
        across = MathNode(tree, location=(22, -27.4), operation="ADD",
                          inputs0=first.std_out, inputs1=along.std_out,
                          name="CursorPosition")
        at = CombineXYZ(tree, location=(23, -27.4), x=across.std_out, y=origin.y,
                        z=origin.z, name="CursorPlace")
        wide = MathNode(tree, location=(21, -28.2), operation="MULTIPLY",
                        inputs0=spacing.std_out, inputs1=self.cursor_width,
                        name="CursorWidth")
        box = Quadrilateral(tree, location=(24, -28.2), mode="RECTANGLE",
                            width=wide.std_out,
                            height=self.cursor_height * self.display_height)
        # a bare curve renders as a hair thin enough to disappear
        wire = CurveWireFrame(tree, location=(25, -28.2), radius=self.frame_radius,
                              resolution=4, geometry=box.geometry_out)
        place = TransformGeometry(tree, location=(26, -28.2), translation=at.std_out,
                                  rotation=[pi / 2, 0, 0], name="PlaceCursor")
        painted = SetMaterial(tree, location=(27, -28.2),
                              material=control["PointerColor"].std_out,
                              name="PaintCursor")
        create_geometry_line(tree, [place, painted], ins=wire.geometry_out)

        frame = Frame(tree, location=(20.6, -26.6), label="CurrentDisplay")
        frame.add([along, across, at, wide, box, wire, place, painted])
        return painted.geometry_out

    # ----------------------------------------------------------------
    # def _create_simulated_geometry_frame(self, tree, control, variables, run):
    #     """``SimulatedGeometry``: everything that is redrawn every frame.
    #
    #     What the machine has printed, written into its box, and the marker
    #     under the cell the head is on. This is built from the state the
    #     simulation zone *outputs*, not inside the zone: none of it is state, it
    #     is a picture of the state.
    #
    #     :return: the geometry socket.
    #     """
    #     label, y = "OutputText", -8
    #     # Plain text centred on the origin, *not* String to Curves' own
    #     # SCALE_TO_FIT: a text box hangs off the origin rather than surrounding
    #     # it, and where inside the box the text ends up moves with how far it
    #     # had to be shrunk - a long string comes out below the box that a two
    #     # letter one sits in the middle of. Centred text is in the same place
    #     # whatever it says, and the fitting is done below, where it can be
    #     # measured.
    #     curves = StringToCurves(tree, location=(26, y), string=run["Output"],
    #                             size=0.6 * self.display_height, align_x="CENTER",
    #                             align_y="MIDDLE", name=label, hide=True)
    #     realize = RealizeInstances(tree, location=(27, y))
    #     fill = FillCurve(tree, location=(28, y), mode="N-gons")
    #     # how much wider than its box the text came out
    #     bounds = BoundingBox(tree, location=(29, y - 1.4))
    #     extent = VectorMath(tree, location=(30, y - 1.4), operation="SUBTRACT",
    #                         inputs0=bounds.max_out, inputs1=bounds.min_out,
    #                         name="Extent" + label)
    #     across = SeparateXYZ(tree, location=(31, y - 1.4), vector=extent.std_out)
    #     # an empty string has no geometry and hence no width; the guard keeps
    #     # the division finite, and the MINIMUM below then leaves it alone at
    #     # scale 1
    #     wide = MathNode(tree, location=(32, y - 1.4), operation="MAXIMUM",
    #                     inputs0=across.x, inputs1=1e-3, name="Width" + label)
    #     ratio = MathNode(tree, location=(33, y - 1.4), operation="DIVIDE",
    #                      inputs0=box_width.std_out, inputs1=wide.std_out,
    #                      name="Ratio" + label)
    #     # only ever shrink: a short output should not be blown up to the full
    #     # width of its box
    #     factor = MathNode(tree, location=(34, y - 1.4), operation="MINIMUM",
    #                       inputs0=ratio.std_out, inputs1=1.0, name="Fit" + label)
    #     scale = CombineXYZ(tree, location=(35, y - 1.4), x=factor.std_out,
    #                        y=factor.std_out, z=factor.std_out, name="Scale" + label)
    #     place = TransformGeometry(tree, location=(29, y), translation=position.std_out,
    #                               rotation=[pi / 2, 0, 0], scale=scale.std_out,
    #                               name="Place" + label)
    #     create_geometry_line(tree, [realize, fill, place], ins=curves.geometry_out)
    #     tree.links.new(fill.geometry_out, bounds.geometry_in)
    #     pieces = [curves, realize, fill, bounds, extent, across, wide, ratio,
    #               factor, scale, place]
    #     written = [place]
    #
    #     # --- the head marker -------------------------------------------
    #     # the x of the cell is read off the tape rather than recomputed from
    #     # TapeSize and CellSize, so the marker cannot drift away from the cells
    #     # if the spacing of the Mesh Line is ever changed
    #     at = Position(tree, location=(26, -14))
    #     spot = SampleIndex(tree, location=(27, -14), data_type="FLOAT_VECTOR",
    #                        domain="POINT", geometry=run["Geometry"], value=at.std_out,
    #                        index=run["PointerPosition"], name="CellPosition")
    #     along = SeparateXYZ(tree, location=(28, -14), vector=spot.std_out)
    #     drop = SeparateXYZ(tree, location=(28, -14.8),
    #                        vector=control["PointerOffset"].std_out)
    #     under = CombineXYZ(tree, location=(29, -14), x=along.x, y=drop.y, z=drop.z,
    #                        name="MarkerPosition")
    #     # an arrow pointing up at the cell, short enough to stay in the gap
    #     # between the tape and the read-outs below it
    #     tip = ConeMesh(tree, location=(26, -16), vertices=32, radius_top=0,
    #                    radius_bottom=0.2 * self.cell_size, depth=0.5 * self.cell_size)
    #     stem = CylinderMesh(tree, location=(26, -17), vertices=32,
    #                         radius=0.1 * self.cell_size, depth=0.5 * self.cell_size)
    #     lowered = TransformGeometry(tree, location=(27, -17),
    #                                 translation=[0, 0, 0],
    #                                 name="StemBelowTip")
    #     create_geometry_line(tree, [stem, lowered])
    #     marker = JoinGeometry(tree, location=(28, -16))
    #     tree.links.new(tip.geometry_out, marker.geometry_in)
    #     tree.links.new(lowered.geometry_out, marker.geometry_in)
    #     put = TransformGeometry(tree, location=(29, -16), translation=under.std_out,
    #                             name="PlaceMarker")
    #     painted = SetMaterial(tree, location=(30, -16),
    #                           material=control["PointerColor"].std_out,
    #                           name="PaintMarker")
    #     create_geometry_line(tree, [marker, put, painted])
    #
    #     # the three strings are painted together, and only then joined with the
    #     # marker: a Set Material without a selection paints everything it is
    #     # handed, so putting the marker in first would take its colour away
    #     lettering = JoinGeometry(tree, location=(33, -10))
    #     for piece in written:
    #         tree.links.new(piece.geometry_out, lettering.geometry_in)
    #     text = SetMaterial(tree, location=(34, -10),
    #                        material=control["GlyphColor"].std_out, name="PaintText")
    #     create_geometry_line(tree, [lettering, text])
    #
    #     # the strip carries its own colours, one per instruction, so it joins
    #     # after the painting rather than before it
    #     strip = self._create_program_strip(tree, control, variables, run)
    #     joined = JoinGeometry(tree, location=(35, -12))
    #     for piece in (text.geometry_out, painted.geometry_out, strip):
    #         tree.links.new(piece, joined.geometry_in)
    #
    #     frame = Frame(tree, location=(25.6, -7.2), label="SimulatedGeometry")
    #     frame.add(pieces + [at, spot, along, drop, under, tip, stem, lowered,
    #                         marker, put, painted, lettering, text, joined])
    #     return joined.geometry_out


class SoupWatcherModifier(GeometryNodesModifier):
    """
    100 tapes from a running ``soup.Soup``, read off ``soup_watcher.py``'s
    data file and laid out flat: two columns of 50, each tape a row of 64
    cells running left to right. Every ``frames_per_snapshot`` frames of the
    scene, the block of 100 tapes on screen is replaced by the next one
    recorded in the file - watching the animation run is watching the soup
    evolve, the same way ``soup_watcher.py`` watched it in the first place.

    See ``BrainFuckSimpleModifier``/``BrainFuckExtendedModifier`` above for
    the general node vocabulary this reuses - ``ForEachZone`` + ``Slice
    String`` to turn text into per-cell geometry, ``make_function`` RPN for
    the layout math, ``FindInStSoupWatcherModifierring`` to test a character against an
    operator's colour. Two things are deliberately different here, though:

    - Nothing stands a glyph upright to face the camera. A cell's character,
      when it has one, is extruded straight out of the flat tape it sits on
      (:class:`ExtrudeMesh`, a fixed ``+z``) rather than counter-rotated to
      stand perpendicular to it - "stick out of the tape" rather than "stand
      next to it". Every tape, background strip and whatever sticks up from
      it alike, is therefore built flat in the x-y plane and shares a single
      tilt at the very end, so a hundred tapes read as one flat, angled
      sheet rather than a hundred separate little upright cards.
    - There is no simulation zone anywhere in this graph, unlike the two BFF
      machines: each frame's picture depends only on the frame number, not
      on the previous frame's, so :class:`SceneTime` reads the current frame
      directly and a couple of ``floor``/``modulo`` RPN steps turn it into
      which snapshot's 100 tape rows of the data file are on screen.

    The whole file is piped in through one ``Import CSV`` node rather than
    baked into the graph. Blender cannot carry text as a geometry attribute -
    there is no ``STRING`` attribute type, so ``Import CSV`` keeps only
    *numeric* columns - so the data is stored as *bytes*: 64 integer columns
    ``c0..c63``, one per cell, and one row per tape (see
    ``video_bff/data/soup_evolution_bytes.csv``). A cell reads its byte by
    sampling the column ``cell`` of the tape at row ``snapshot_offset +
    tapeIdx`` (an ``Index Switch`` picks the column, since a named attribute
    cannot be chosen by a runtime index), turns the byte back into a one-
    character string with a ``Slice String`` into a 128-entry ascii table,
    and hands that to exactly the same colour/curve pipeline the string once
    fed. This keeps the ~6 M cells out of the ``.blend`` - a baked string
    that big is what an earlier version choked on.

    A cell shows nothing unless its byte was one of the ten BFF instructions
    when ``soup_watcher.py`` recorded it: that is what ``soup.render()``
    already encoded (an instruction's own character, ``'0'`` for a zero
    byte, ``' '`` for anything else), and the ascii table reproduces the same
    character from the byte, so a blank cell here is a cell ``Switch`` never
    lets through to ``String to Curves`` at all, not a cell drawn and then
    hidden.

    :param data_file: the byte csv (64 columns ``c0..c63``, ``;`` delimited,
        one row per tape) - convert ``soup_watcher.py``'s ``soup_evolution.csv``
        with ``video_bff/data``'s converter. Resolved against ``DATA_DIR``
        unless it is already an absolute path
    :param max_snapshots: use only the first this many snapshots of the file
        (``None`` - the default - uses all of them)
    :param cell_size: width of one cell, and so the spacing of the 64
        characters along a tape
    :param column_gap: gap between the two columns of tapes, in the same
        units as ``cell_size``
    :param row_spacing: vertical distance between one tape and the next
    :param tape_tilt: angle the whole sheet of tapes is tilted back by, so
        that a camera looking along +y sees their faces rather than their
        edges - the "slightly angled view"
    :param glyph_size: height of an instruction's glyph, as a fraction of
        ``cell_size``
    :param stick_out: how far a glyph is extruded above the tape, in the
        same units as ``cell_size``
    :param cell_square: side of one cell's background square, as a fraction of
        ``cell_size`` - just under 1 leaves a thin gap between neighbouring
        cells, so a tape reads as 64 separate squares rather than a solid bar
    :param frames_per_snapshot: how many frames a block of 100 tapes stays
        on screen before the next one takes its place
    :param colors: optional ``{node name: colour name}`` overriding the
        colour of an instruction (see :data:`INSTRUCTION_COLORS`),
        or the two entries ``GlyphColor`` (an instruction's default colour,
        never actually seen - every glyph on screen is one of the ten and so
        always overridden by its own colour, but it is the fall-back that
        keeps that chain total) and ``TapeColor`` (the background strips).
    """

    TAPE = 64  # bytes per tape - matches soup.py's TAPE and soup_watcher.py's data
    ROWS = 50  # tapes stacked in one column
    COLUMNS = 2
    TAPES_PER_SNAPSHOT = ROWS * COLUMNS  # 100, matching soup_watcher.py's --top

    # the ten instructions and their colours - the same list
    # BrainFuckExtendedModifier colours its own tapes and code table with, so
    # a soup tape and a running machine's tape read as the same alphabet
    OPCODE_COLORS = INSTRUCTION_COLORS
    OPERATORS = BFFNode.COMMANDS
    GLYPH_COLOR = "text"
    TAPE_COLOR = "gray_1"

    def __init__(self, data_file="soup_evolution_bytes.csv", max_snapshots=None,
                 cell_size=0.09, column_gap=0.4, row_spacing=0.13,
                 tape_tilt=0, glyph_size=0.85, stick_out=0.05, cell_square=0.9,
                 frames_per_snapshot=10, colors=None, name="SoupWatcher",
                 **kwargs):
        self.cell_size = cell_size
        self.column_gap = column_gap
        self.row_spacing = row_spacing
        self.tape_tilt = tape_tilt
        self.glyph_size = glyph_size
        self.stick_out = stick_out
        self.cell_square = cell_square
        self.frames_per_snapshot = frames_per_snapshot

        overrides = colors or {}
        self.opcode_colors = tuple((node_name, overrides.get(node_name, color), character)
                                   for node_name, color, character in self.OPCODE_COLORS)
        self.glyph_color = overrides.get("GlyphColor", self.GLYPH_COLOR)
        self.tape_color = overrides.get("TapeColor", self.TAPE_COLOR)

        path = data_file if os.path.isabs(data_file) else os.path.join(DATA_DIR, data_file)
        self.data_file = path
        self.num_snapshots = self._count_snapshots(path, max_snapshots)

        self.kwargs = kwargs
        super().__init__(name=name, automatic_layout=False)

    # ----------------------------------------------------------------
    @classmethod
    def _count_snapshots(cls, path, max_snapshots):
        """How many whole snapshots of 100 tapes the byte csv holds.

        The data itself never passes through python - ``Import CSV`` reads it
        at render time (see :meth:`_create_control_frame`). All that is needed
        here is the row count, so the ``... mod NumSnapshots`` wrap in
        :meth:`_create_snapshot_offset` knows where the file ends. Rows are
        counted by streaming the file (one tape per line, plus a header line),
        never holding it in memory - the file is ~20 MB of integers.

        :return: the number of whole snapshots (blocks of
            :attr:`TAPES_PER_SNAPSHOT` tapes), capped to ``max_snapshots``
            when that is given.
        """
        with open(path, newline="") as file:
            tape_rows = sum(1 for line in file if line.strip()) - 1  # minus header

        num_snapshots = tape_rows // cls.TAPES_PER_SNAPSHOT
        if max_snapshots is not None:
            num_snapshots = min(num_snapshots, max_snapshots)
        if num_snapshots < 1:
            raise ValueError("%s has fewer than %d tapes (one snapshot)"
                             % (path, cls.TAPES_PER_SNAPSHOT))
        return num_snapshots

    # ----------------------------------------------------------------
    @staticmethod
    def _ascii_table():
        """A 128-character string whose position ``i`` is ``chr(i)``.

        Non-printables (below 32, and 127) are blanked so a stray byte can
        never smuggle a newline or control code into a glyph; every character
        the tapes actually carry - the ten instructions, ``'0'`` and ``' '`` -
        is printable and lands on itself. ``Slice String`` at position
        ``byte`` then reverses ``ord`` the byte csv did.
        """
        return "".join(chr(i) if 32 <= i < 127 else " " for i in range(128))

    # ----------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        control = self._create_control_frame(tree)
        offset = self._create_snapshot_offset(tree, control)
        bars = self._create_tape_bars_frame(tree, control)
        glyphs = self._create_glyphs_frame(tree, control, offset)

        joined = JoinGeometry(tree, location=(10, 0))
        tree.links.new(bars, joined.geometry_in)
        tree.links.new(glyphs, joined.geometry_in)
        # everything above is flat in the x-y plane; one tilt at the very end
        # is what makes this "a slightly angled view" rather than a camera
        # looking at a hundred tapes edge-on
        tilt = TransformGeometry(tree, location=(11, 0), rotation=[self.tape_tilt, 0, 0],
                                 name="TiltIntoView")
        create_geometry_line(tree, [joined, tilt])

        out = self.group_outputs
        tree.links.new(tilt.geometry_out, out.inputs["Geometry"])

    # ----------------------------------------------------------------
    def _create_control_frame(self, tree):
        """``Control``: the data, the palette, and the two knobs that drive time.

        :return: ``{name: node}``, so that the frames downstream can pick
            what they need by the name it carries in the editor.
        """
        x = -14
        control = {
            "CellSize": InputValue(tree, location=(x, 0), value=self.cell_size,
                                   name="CellSize", hide=True),
            "FramesPerSnapshot": InputInteger(tree, location=(x, -0.6),
                                              integer=self.frames_per_snapshot,
                                              name="FramesPerSnapshot", hide=True),
            "NumSnapshots": InputInteger(tree, location=(x, -1.2),
                                         integer=self.num_snapshots,
                                         name="NumSnapshots", hide=True),
            # the whole data file, read at render time as a point cloud of one
            # point per tape with 64 INT columns ``c0..c63`` - one per cell.
            # ``;`` is the delimiter because the file's integers are separated
            # by it and it never appears otherwise. The glyph frame samples a
            # cell's column of the right tape row out of this (see
            # :meth:`_create_byte_lookup`); nothing baked, unlike the string
            # this replaced.
            "Csv": ImportCSV(tree, location=(x, -1.8), delimiter=";",
                             path=self.data_file, name="Csv", label="soup bytes",
                             hide=True),
            # position i holds ``chr(i)`` (non-printables blanked), so that a
            # byte sampled above becomes its character with one Slice String -
            # the byte was that character's ascii code when the file was written
            "AsciiTable": InputString(tree, location=(x, -2.4),
                                      string=self._ascii_table(),
                                      name="AsciiTable", hide=True),
            "Operators": InputString(tree, location=(x, -3.0), string=self.OPERATORS,
                                     name="Operators", hide=True),
        }

        # **self.kwargs carries things like `emission=0.6` through to every
        # material - see BrainFuckSimpleModifier._create_control_frame's
        # identical forwarding. These scenes are lit mostly by emission on a
        # black background (video_bff/scene_bff.py's `_setup_render`), and
        # the plain preset materials render nearly black without it.
        palette = {}
        for row, (node_name, color, _) in enumerate(self.opcode_colors):
            palette[node_name] = InputMaterial(tree, location=(x, -3.4 - 0.4 * row),
                                               material=color, name=node_name,
                                               **self.kwargs, hide=True)
        palette["GlyphColor"] = InputMaterial(
            tree, location=(x, -3.4 - 0.4 * len(self.opcode_colors)),
            material=self.glyph_color, name="GlyphColor", **self.kwargs, hide=True)
        palette["TapeColor"] = InputMaterial(
            tree, location=(x, -3.8 - 0.4 * len(self.opcode_colors)),
            material=self.tape_color, name="TapeColor", **self.kwargs, hide=True)
        for source in palette.values():
            self.materials.append(source.node.material)
        control.update(palette)

        # the ten instruction colours travel as one bundle - see the same
        # choice in BrainFuckExtendedModifier._create_control_frame
        control["OpColors"] = CombineBundle(
            tree, location=(x + 1.6, -3.4), name="OPColorBundle",
            items=[(node_name, "MATERIAL", palette[node_name].std_out)
                   for node_name, _, _ in self.opcode_colors])

        frame = Frame(tree, location=(x - 0.4, 0.6), label="Control")
        frame.add(list(control.values()))
        return control

    # ----------------------------------------------------------------
    def _create_snapshot_offset(self, tree, control):
        """Which tape row the block on screen right now starts at.

        A pure function of the current frame - no simulation zone needed,
        since nothing here accumulates: the block index is
        ``floor(frame / FramesPerSnapshot) mod NumSnapshots``, and the offset
        is that many whole snapshots' worth of tape *rows* into the point
        cloud (:attr:`TAPES_PER_SNAPSHOT` rows each).

        :return: an INT socket, the tape-row offset of the current snapshot.
        """
        frame_now = SceneTime(tree, location=(-8, 3), std_out="Frame", hide=True)
        offset = make_function(
            tree, name="SnapshotOffset",
            functions={
                "offset": "frame,fps,/,floor,n,%%,%d,*" % self.TAPES_PER_SNAPSHOT
            },
            inputs=["frame", "fps", "n"], outputs=["offset"],
            scalars=["frame", "fps"], integers=["n", "offset"],
            hide=True, location=(-7, 3))
        tree.links.new(frame_now.std_out, offset.inputs["frame"])
        tree.links.new(control["FramesPerSnapshot"].std_out, offset.inputs["fps"])
        tree.links.new(control["NumSnapshots"].std_out, offset.inputs["n"])

        frame = Frame(tree, location=(-8.2, 3.6), label="SnapshotOffset")
        frame.add([frame_now, offset])
        return offset.outputs["offset"]

    # ----------------------------------------------------------------
    def _create_tape_bars_frame(self, tree, control):
        """``TapeBars``: 64 little cell squares per tape, not one long strip.

        Each tape is a ``MeshLine`` of :attr:`TAPE` points - one per cell,
        ``cell_size`` apart - and a small square (``cell_square`` of a cell
        wide, so a thin gap shows between neighbours) is instanced on every
        point. That row of 64 is then instanced onto one point per tape,
        placed by column and row exactly as :meth:`_create_glyphs_frame`
        places that tape's cells, so a glyph always sits on its own square.

        :return: the geometry socket of the 100 x 64 squares.
        """
        n = self.TAPES_PER_SNAPSHOT
        cell_size = control["CellSize"].std_out

        # one tape's 64 cell centres: a mesh line from x=0 to (TAPE-1)*cellSize,
        # so cell c sits at c*cellSize - the same x a glyph's cell uses
        span = MathNode(tree, location=(-7, -3.4), operation="MULTIPLY",
                        inputs0=cell_size, inputs1=self.TAPE - 1,
                        name="TapeSpan", hide=True)
        end = CombineXYZ(tree, location=(-6.4, -3.4), x=span.std_out, name="TapeEnd",
                         hide=True)
        cell_line = MeshLine(tree, location=(-6, -3.2), mode="END_POINTS",
                             count=self.TAPE, start_location=Vector([0, 0, 0]),
                             end_location=end.std_out)

        # the square instanced on each cell centre, a hair under a cell wide so
        # neighbours do not touch - the "small separation" between cells
        side = MathNode(tree, location=(-7, -4), operation="MULTIPLY",
                        inputs0=cell_size, inputs1=self.cell_square,
                        name="CellSquareSide", hide=True)
        square = Quadrilateral(tree, location=(-6, -4), mode="RECTANGLE",
                               width=side.std_out, height=side.std_out)
        fill = FillCurve(tree, location=(-5, -4), mode="N-gons")
        create_geometry_line(tree, [square, fill])
        cells = InstanceOnPoints(tree, location=(-4.4, -3.2), points=cell_line.geometry_out,
                                 instance=fill.geometry_out, name="CellSquares")

        # one point per tape, placed at its column/row - the glyph layout
        # without the per-cell term, which the mesh line above supplies
        index = Index(tree, location=(-8, -2), hide=True)
        layout = make_function(
            tree, name="TapeRowPosition",
            aux_functions={
                "row": "index,%d,%%" % self.ROWS,
                "col": "index,%d,/,floor" % self.ROWS,
            },
            functions={
                "position": [
                    "col,%d,cellSize,*,columnGap,+,*" % self.TAPE,
                    "row,rowSpacing,*",
                    "0",
                ],
            },
            inputs=["index", "cellSize", "columnGap", "rowSpacing"],
            outputs=["position"], vectors=["position"],
            scalars=["index", "cellSize", "columnGap", "rowSpacing", "row", "col"],
            hide=True, location=(-7, -2))
        tree.links.new(index.std_out, layout.inputs["index"])
        tree.links.new(cell_size, layout.inputs["cellSize"])
        layout.inputs["columnGap"].default_value = self.column_gap
        layout.inputs["rowSpacing"].default_value = self.row_spacing

        points = Points(tree, location=(-6, -2), count=n)
        placed = SetPosition(tree, location=(-5, -2), position=layout.outputs["position"])
        create_geometry_line(tree, [points, placed])

        # each tape point carries a whole 64-square row; realize flattens both
        # instance levels into the 6400 squares that make up the sheet
        tapes = InstanceOnPoints(tree, location=(-3, -2), points=placed.geometry_out,
                                 instance=cells.geometry_out, name="TapeRows")
        realize = RealizeInstances(tree, location=(-2, -2))
        painted = SetMaterial(tree, location=(-1, -2), material=control["TapeColor"].std_out,
                              name="PaintTapes")
        create_geometry_line(tree, [tapes, realize, painted])

        frame = Frame(tree, location=(-8.2, -1.4), label="TapeBars")
        frame.add([index, layout, points, placed, span, end, cell_line, side, square,
                   fill, cells, tapes, realize, painted])
        return painted.geometry_out

    # ----------------------------------------------------------------
    def _create_byte_lookup(self, tree, control, cell_index, snapshot_offset):
        """The byte on screen at each display cell.

        ``cell_index`` is the flat index of the display cell
        (``0 .. TAPES_PER_SNAPSHOT*TAPE - 1``). ``cell = index mod TAPE`` is
        which of the 64 columns it wants, ``tapeIdx = index // TAPE`` which of
        the snapshot's 100 tapes it is on, and that tape's row in the point
        cloud is ``snapshot_offset + tapeIdx``.

        A named attribute cannot be chosen by a value known only at render
        time, so all 64 columns ``c0..c63`` are sampled at that row and an
        ``Index Switch`` keyed on ``cell`` keeps the one that belongs here -
        the byte that was this cell's character's ascii code.

        :return: ``(switch, nodes)`` - the Index Switch whose ``std_out`` is
            the byte, and every node built here so the caller can frame them.
        """
        cell = IntegerMath(tree, location=(-6, -10.0), operation="MODULO",
                           inputs0=cell_index, inputs1=self.TAPE,
                           name="CellInTape", hide=True)
        tape_idx = IntegerMath(tree, location=(-6, -10.3), operation="DIVIDE",
                               inputs0=cell_index, inputs1=self.TAPE,
                               name="TapeIndex", hide=True)
        row = IntegerMath(tree, location=(-6, -10.6), operation="ADD",
                          inputs0=tape_idx.std_out, inputs1=snapshot_offset,
                          name="TapeRow", hide=True)

        csv_geo = control["Csv"].geometry_out
        switch = IndexSwitch(tree, location=(-3.4, -10.3), data_type="INT",
                             index=cell.std_out, name="CellByte", hide=True)
        nodes = [cell, tape_idx, row, switch]
        for c in range(self.TAPE):
            attr = NamedAttribute(tree, location=(-5.2, -10.0 - 0.04 * c),
                                  data_type="INT", name="c%d" % c, label="c%d" % c,
                                  hide=True)
            sample = SampleIndex(tree, location=(-4.4, -10.0 - 0.04 * c),
                                 data_type="INT", domain="POINT", geometry=csv_geo,
                                 value=attr.std_out, index=row.std_out, hide=True)
            switch.add_item(socket=sample.std_out)
            nodes += [attr, sample]
        return switch, nodes

    # ----------------------------------------------------------------
    def _create_glyphs_frame(self, tree, control, snapshot_offset):
        """``Glyphs``: the operator that shows through each cell, if any.

        One point per cell of every tape in the snapshot -
        :attr:`TAPES_PER_SNAPSHOT` times :attr:`TAPE` of them, laid out the
        same way :meth:`_create_tape_bars_frame` lays out the tapes
        themselves - then :meth:`_create_byte_lookup` reads the byte at this
        cell out of the imported point cloud, a ``ForEachZone`` turns it back
        into its character, keeps it only if it is one of the ten instructions
        (blank otherwise), and turns what is left into a letter standing
        extruded out of the tape rather than lying flat on it.

        :return: the geometry socket of however many glyphs the snapshot holds.
        """
        count = self.TAPES_PER_SNAPSHOT * self.TAPE
        index = Index(tree, location=(-8, -8), hide=True)
        layout = make_function(
            tree, name="CellPosition",
            aux_functions={
                "cell": "index,%d,%%" % self.TAPE,
                "tapeIdx": "index,%d,/,floor" % self.TAPE,
                "row": "tapeIdx,%d,%%" % self.ROWS,
                "col": "tapeIdx,%d,/,floor" % self.ROWS,
            },
            functions={
                "position": [
                    "cell,cellSize,*,col,%d,cellSize,*,columnGap,+,*,+" % self.TAPE,
                    "row,rowSpacing,*",
                    "0",
                ],
            },
            inputs=["index", "cellSize", "columnGap", "rowSpacing"],
            outputs=["position"], vectors=["position"],
            scalars=["index", "cellSize", "columnGap", "rowSpacing",
                     "cell", "tapeIdx", "row", "col"],
            hide=True, location=(-7, -8))
        tree.links.new(index.std_out, layout.inputs["index"])
        tree.links.new(control["CellSize"].std_out, layout.inputs["cellSize"])
        layout.inputs["columnGap"].default_value = self.column_gap
        layout.inputs["rowSpacing"].default_value = self.row_spacing

        points = Points(tree, location=(-6, -8), count=count)
        placed = SetPosition(tree, location=(-5, -8), position=layout.outputs["position"])
        create_geometry_line(tree, [points, placed])

        byte, byte_nodes = self._create_byte_lookup(tree, control, index.std_out,
                                                    snapshot_offset)
        position = Position(tree, location=(-7, -9.6), hide=True)

        zone = ForEachZone(tree, location=(-4, -8), domain="POINT", node_width=9,
                           geometry=placed.geometry_out)
        zone.add_socket(socket_type="INT", name="Byte",
                        value=byte.std_out, for_input=True)
        glyph_shift = VectorMath(tree, location=(-6, -9.6), label="GlyphShift", inputs0=position.std_out,
                                 inputs1=Vector([0, 0, 0.1]), hide=True)
        zone.add_socket(socket_type="VECTOR", name="Location",
                        value=glyph_shift.std_out, for_input=True)

        # the byte is this cell's character's ascii code (what the byte csv
        # stored); one Slice String into the ascii table turns it back into the
        # character, and everything downstream is exactly what the string
        # version fed - a Find in String colour test and String to Curves.
        letter = SliceString(tree, location=(-3, -8), string=control["AsciiTable"].std_out,
                             position=zone.foreach_input.outputs["Byte"],
                             length=1, name="Letter")

        # one boolean per colour of the shared palette, plus the one that says
        # the byte is an instruction at all - see instruction_selector
        socket_labels = [node_name for node_name, _, _ in self.opcode_colors]
        color_selection = instruction_selector(
            tree, letter.std_out, self.opcode_colors, location=(-3, -9.4),
            commands=control["Operators"].std_out)
        is_operator = color_selection.outputs["IsOperator"]

        # a cell that is not one of the ten instructions gets the empty
        # string, and String to Curves draws nothing for it - a blank cell
        # is one that never reaches the curve at all, not one drawn and
        # then hidden
        glyph = Switch(tree, location=(-2, -8), input_type="STRING", switch=is_operator,
                       true=letter.std_out, false="", name="GlyphOrBlank")

        size = MathNode(tree, location=(-2, -8.6), operation="MULTIPLY",
                        inputs0=control["CellSize"].std_out, inputs1=self.glyph_size,
                        name="GlyphSize", hide=True)
        curves = StringToCurves(tree, location=(-1, -8), string=glyph.std_out,
                                size=size.std_out, align_x="CENTER", align_y="MIDDLE")
        realize = RealizeInstances(tree, location=(0, -8))
        fill = FillCurve(tree, location=(1, -8), mode="N-gons")
        # the tape is flat in x-y; a fixed +z offset is what "sticking out of
        # the tape" means here, rather than an extrusion along whatever the
        # fill's own face normal happens to be
        stick_out = ExtrudeMesh(tree, location=(2, -8), mode="FACES",
                                offset=Vector([0, 0, 1]), offset_scale=self.stick_out,
                                name="StickOut")

        opcolors = SeparateBundle(tree, location=(1, -10), bundle=control["OpColors"].std_out,
                                  items=control["OpColors"].items, name="OPColors")
        painters = ([SetMaterial(tree, location=(3, -8), material=control["GlyphColor"].std_out,
                                 name="PaintDefault")]
                    + [SetMaterial(tree, location=(3, -8.4 - 0.3 * row),
                                   selection=color_selection.outputs[node_name],
                                   material=opcolors.out(node_name),
                                   name="Paint" + node_name, hide=True)
                       for row, node_name in enumerate(socket_labels)])

        placed_glyph = TransformGeometry(tree, location=(4, -8), rotation=Vector([pi / 2, 0, 0]),
                                         translation=zone.foreach_input.outputs["Location"],
                                         name="PlaceGlyph")
        zone.create_geometry_line([realize, fill, stick_out] + painters + [placed_glyph],
                                  ins=curves.geometry_out)

        frame = Frame(tree, location=(-8.2, -7.4), label="Glyphs")
        frame.add([index, layout, points, placed, position, zone, letter,
                   color_selection, glyph, size, curves, realize, fill, stick_out,
                   opcolors, placed_glyph] + byte_nodes + painters)
        return zone.geometry_out


class SoupWatcherModifierSingle(SoupWatcherModifier):
    """The same snapshot of 100 tapes, stacked as **one** column instead of two.

    Same data file, same snapshot clock, same cells and glyphs - the only
    thing that changes is where a tape goes: with all
    :attr:`TAPES_PER_SNAPSHOT` tapes in a single column, ``col`` is always 0
    and the layout collapses to "tape *i* is the *i*-th row". ``column_gap``
    is inherited but no longer reaches the geometry.

    The point of the arrangement is that one tape can then be read. Two
    columns of 50 have to share the frame width between them, which leaves
    each tape half a screen wide and its 64 cells too small to make out; a
    single column lets the near tape run the full width of the frame, with
    the rest of the snapshot receding above it and the far end of the column
    running off the top of the screen. Which of the 100 rows still fit is a
    property of the shot, not of this graph - see
    ``video_bff/scene_bff.py``'s ``soup_watcher``, which places the camera by
    fitting the *first* row edge to edge and lets the rest fall where they
    fall.

    Because the tilt lives in the camera rather than in the geometry (the
    default ``tape_tilt`` is 0 and the sheet is built flat in x-y), nothing
    about the angle of the view is decided here either.
    """

    ROWS = SoupWatcherModifier.TAPES_PER_SNAPSHOT  # the whole snapshot, stacked
    COLUMNS = 1
    TAPES_PER_SNAPSHOT = ROWS * COLUMNS

    def __init__(self, name="SoupWatcherSingle", **kwargs):
        super().__init__(name=name, **kwargs)

    @property
    def tape_width(self):
        """How wide one tape is, in the units the layout is built in.

        The 64 cell *centres* span ``(TAPE - 1) * cell_size`` and each end
        cell adds half its square, so the strip a camera has to frame is one
        whole cell wider than the distance between the outermost centres.
        """
        return self.TAPE * self.cell_size

    @property
    def tape_center(self):
        """The x a camera has to sit above to see a tape centred.

        Cell 0 is at ``x = 0`` and cell 63 at ``x = 63 * cell_size``, so the
        middle of the strip is half a cell to the left of half its width.
        """
        return (self.TAPE - 1) * self.cell_size / 2


class SoupWatcherModifierSingleStarWars(SoupWatcherModifierSingle):
    """The single-column soup, plus an end card that crawls out of it.

    Everything :class:`SoupWatcherModifierSingle` builds, with two extra
    dials for the scene to turn once the soup has been watched long enough:

    ``Recede``
        pushes the whole sheet of tapes away *along the line of sight*, so
        it shrinks towards the middle of the frame and settles behind
        whatever comes next without sliding sideways out of the shot. That
        is what "the tapes shift to the background" means here.
    ``CrawlDistance``
        how far the end card has travelled up its own plane. At 0 the text
        sits at the near edge of the tape sheet; the scene runs it from
        somewhere below the bottom of the frame to well up the screen.

    The card is the receding-title-crawl trick: a block of text lying in a
    plane that is tilted **away** from the camera, so that moving it up the
    plane also moves it away and perspective alone shrinks it towards a
    vanishing point.

    The tilt is measured from the tapes' own plane, and what matters is what
    it leaves between the card and the line of sight. The shot's camera looks
    down on the tapes at about 41 degrees, so the 15 this defaults to puts
    the card some 26 degrees off the line of sight: compressed hard, with the
    vanishing point landing within a whisker of the top edge of the frame.
    Lay the card flat in the tapes' plane instead (``crawl_tilt=0``) and it
    merely slides away without shrinking much; tilt it much further and the
    vanishing point drops into the frame, where the lettering piles up on
    itself and never leaves. A camera at a different angle wants a different
    tilt - the two add up.

    Because the card's plane sinks below the tapes as it recedes (that is
    what tilting away *means*), the two would intersect if the sheet stayed
    put - which is the other reason ``Recede`` exists, and why the scene
    pushes the tapes back before the crawl arrives rather than after.

    The text is set in ``crawl_font``, which has to be one of the fonts
    ``perform.scene.initialize_blender`` loads. Of those, Arial Black is the
    heavy grotesque closest in weight and feel to the lettering this kind of
    crawl is usually set in.

    :param crawl_text: the card, newlines and all - ``String to Curves``
        breaks lines on ``\\n`` and centres them.
    :param crawl_font: name of a loaded Blender font.
    :param crawl_color: palette name for the lettering.
    :param crawl_size: cap height of the lettering, in the same units as
        ``cell_size``; the widest line has to fit the frame at the near end
        of the crawl, which is what sets it.
    :param crawl_tilt: how far the card's plane is tilted away from the
        plane of the tapes, in radians.
    :param crawl_lift: how far above the tapes the card starts, so that it
        clears them at ``CrawlDistance = 0``.
    :param line_spacing: multiples of the line height between the lines.
    :param view_direction: the direction the camera looks, which is the
        direction ``Recede`` pushes the tapes along. Only its direction is
        used; the scene's tilt and this have to agree, which is why both
        take the same three numbers.
    """

    CRAWL_TEXT = ("Computational Life Generator\n"
                  "by Alex Borger\n"
                  "https://alexborger.com/clr-computational-life-reactor")
    CRAWL_FONT = "Arial Black"  # loaded by perform.scene.initialize_blender
    CRAWL_COLOR = "example"

    # the widest line of the default card is 20.6 units long set solid in
    # Arial Black, and the frame is about 5.5 wide where the crawl starts
    def __init__(self, crawl_text=None, crawl_font=None, crawl_color=None,
                 crawl_size=0.25, crawl_tilt=15 * pi / 180, crawl_lift=0.35,
                 line_spacing=1.6, view_direction=(0, 3.9, -3.4),
                 name="SoupWatcherStarWars", **kwargs):
        self.crawl_text = self.CRAWL_TEXT if crawl_text is None else crawl_text
        self.crawl_font = crawl_font or self.CRAWL_FONT
        self.crawl_color = crawl_color or self.CRAWL_COLOR
        self.crawl_size = crawl_size
        self.crawl_tilt = crawl_tilt
        self.crawl_lift = crawl_lift
        self.line_spacing = line_spacing
        self.view_direction = Vector(view_direction).normalized()
        super().__init__(name=name, **kwargs)

    # ----------------------------------------------------------------
    @property
    def crawl_direction(self):
        """Up the crawl: one unit of ``CrawlDistance`` as a world vector.

        The tapes' own "away from the camera" is ``+y``; tilting the card's
        plane back by ``crawl_tilt`` about x tips that direction downwards by
        the same angle.
        """
        return Vector((0, math.cos(self.crawl_tilt), -math.sin(self.crawl_tilt)))

    @property
    def crawl_origin(self):
        """Where the card sits at ``CrawlDistance = 0``.

        Centred on the tape, level with its nearest row and ``crawl_lift``
        above it - the hinge the whole crawl swings out of, and the frame
        anything else riding the crawl (the screenshot, say) has to be
        placed in.
        """
        return Vector((self.tape_center, 0, self.crawl_lift))

    def crawl_position(self, distance):
        """Where a thing riding the crawl at ``distance`` ends up."""
        return self.crawl_origin + distance * self.crawl_direction

    # ----------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        control = self._create_control_frame(tree)
        offset = self._create_snapshot_offset(tree, control)
        bars = self._create_tape_bars_frame(tree, control)
        glyphs = self._create_glyphs_frame(tree, control, offset)

        joined = JoinGeometry(tree, location=(10, 0))
        tree.links.new(bars, joined.geometry_in)
        tree.links.new(glyphs, joined.geometry_in)
        tilt = TransformGeometry(tree, location=(11, 0), rotation=[self.tape_tilt, 0, 0],
                                 name="TiltIntoView")
        recede = self._create_recede_frame(tree)
        create_geometry_line(tree, [joined, tilt, recede])

        # the card is not tilted with the tapes and not pushed back with
        # them either - it has its own plane and its own dial
        crawl = self._create_crawl_frame(tree)

        everything = JoinGeometry(tree, location=(15, 0))
        tree.links.new(recede.geometry_out, everything.geometry_in)
        tree.links.new(crawl, everything.geometry_in)

        out = self.group_outputs
        tree.links.new(everything.geometry_out, out.inputs["Geometry"])

    # ----------------------------------------------------------------
    def _create_recede_frame(self, tree):
        """``Recede``: push the tapes away along the line of sight.

        Along the *view* direction rather than simply backwards, because a
        sheet that fills the frame edge to edge has to shrink towards the
        middle of the frame to read as "further away" - pushing it along +y
        instead would slide it up out of the top of the shot.

        :return: the ``Transform Geometry`` the tapes leave through.
        """
        # only this node may carry "Recede" in its name: the scene looks its
        # dials up by substring (``ibpy.get_geometry_node_from_modifier``) and
        # would otherwise keyframe whichever of them the iteration reached first
        amount = InputValue(tree, location=(11, 1.6), value=0, name="Recede", hide=True)
        d = self.view_direction
        push = make_function(
            tree, name="PushOffset",
            functions={"translation": ["%.6f,r,*" % d.x, "%.6f,r,*" % d.y,
                                       "%.6f,r,*" % d.z]},
            inputs=["r"], outputs=["translation"],
            scalars=["r"], vectors=["translation"], hide=True, location=(12, 1.6))
        tree.links.new(amount.std_out, push.inputs["r"])

        recede = TransformGeometry(tree, location=(13, 0),
                                   translation=push.outputs["translation"],
                                   name="PushBack")
        frame = Frame(tree, location=(10.8, 2.2), label="PushBack")
        frame.add([amount, push, recede])
        return recede

    # ----------------------------------------------------------------
    def _create_crawl_frame(self, tree):
        """``Crawl``: the end card, lying in its own tilted plane.

        One ``String to Curves`` for the whole card - it breaks the lines
        itself and centres them - filled flat rather than extruded: the card
        is a piece of lettering seen in perspective, not an object standing
        in the scene, and anything sticking out of its plane would give that
        away as it recedes.

        :return: the geometry socket of the placed card.
        """
        color = InputMaterial(tree, location=(10, 4.6), material=self.crawl_color,
                              name="CrawlColor", **self.kwargs, hide=True)
        self.materials.append(color.node.material)

        text = InputString(tree, location=(10, 4), string=self.crawl_text,
                           name="CrawlText", label="crawl", hide=True)
        curves = StringToCurves(tree, location=(11, 4), string=text.std_out,
                                font=self.crawl_font, size=self.crawl_size,
                                line_spacing=self.line_spacing,
                                align_x="CENTER", align_y="MIDDLE")
        realize = RealizeInstances(tree, location=(12, 4))
        fill = FillCurve(tree, location=(13, 4), mode="N-gons")
        painted = SetMaterial(tree, location=(14, 4), material=color.std_out,
                              name="PaintCrawl")

        distance = InputValue(tree, location=(10, 3.4), value=0,
                              name="CrawlDistance", hide=True)
        o, m = self.crawl_origin, self.crawl_direction
        move = make_function(
            tree, name="CrawlPosition",
            functions={"translation": ["%.6f" % o.x,
                                       "%.6f,s,%.6f,*,+" % (o.y, m.y),
                                       "%.6f,s,%.6f,*,+" % (o.z, m.z)]},
            inputs=["s"], outputs=["translation"],
            scalars=["s"], vectors=["translation"], hide=True, location=(11, 3.4))
        tree.links.new(distance.std_out, move.inputs["s"])

        place = TransformGeometry(tree, location=(15, 4),
                                  rotation=[-self.crawl_tilt, 0, 0],
                                  translation=move.outputs["translation"],
                                  name="PlaceCrawl")
        create_geometry_line(tree, [curves, realize, fill, painted, place])

        frame = Frame(tree, location=(9.8, 5.2), label="Crawl")
        frame.add([color, text, curves, realize, fill, painted, distance, move, place])
        return place.geometry_out


# One grid unit is 200 px in both directions, so a half-integer coordinate is
# still a round 100 px and nodes cannot end up a few pixels out of line.
GRID = 200

# ---------------------------------------------------------------------------
# The flight path
# ---------------------------------------------------------------------------
# The molecule does not sit still and let the camera fly past it; the camera is
# fixed and the molecule flies. It does that by sliding along a track - a fixed
# space curve - so that point *i* of the molecule sits at arc length
# ``HeadOffset - i * Spacing`` along it. Animating ``HeadOffset`` moves the
# whole thing, and the shape it takes on the way is the shape of the track.
#
# This replaces the sine that used to modulate the axis. A sine could bend the
# molecule but never loop it: a loop is not a function of x, it comes back over
# itself. A track can be any curve at all, and the price is only that it has to
# be computed here and handed to the graph rather than written as one formula.
#
# The track is written to a csv and read back by an Import CSV node, the same
# way the tape scenes get their data. It is regenerated whenever the geometry
# constants below change.
#
# The screen, for reference: the camera sits on -y looking at the x-z plane, so
# x runs across and z runs up. With a 40 mm lens 36 units back the frame is
# about x in [-16.2, 16.2] and z in [-9.1, 9.1].

DNA_TRACK_FILE = "dna_track"

TRACK_X_IN = 15.0  # the track starts here, well off screen to the right
TRACK_Z_UP = -3  # height of the incoming run, in the upper half
TRACK_X_LOOP = 1.0  # where the roller-coaster loop sits
TRACK_R1 = 4.5  # and how big it is
TRACK_DEPTH = 4.5  # how far back in y a loop steps to clear its own entry
TRACK_X_LEFT = -11.0  # how far left it gets before turning back
TRACK_R2 = 4  # radius of the 180 degree turn that sends it back right
TRACK_X_OUT = 30.0  # the track ends here, off screen to the right again
TRACK_STEP = 0.08  # spacing of the samples written to the csv

# Height of the last straight, and so of the fork: this is the one number in
# the track that the *choreography* depends on rather than the composition.
# Everything the molecule does after it unzips happens here - the strand that
# stays runs along this line, and the strand that leaves climbs off the top of
# the frame from it - so how far the peel has to lift, and how far the molecule
# is stretched while it does, is set by how low this is. It was -7, a whole
# frame height below the middle, which cost twenty units of lift; near the
# middle it costs ten.
TRACK_Z_OUT = -3.5
# and therefore the height the turn has to start from, since a 180 degree turn
# drops exactly its own diameter. Derived rather than written down, because
# what matters is where the turn comes *out*.
TRACK_Z_MID = TRACK_Z_OUT + 2.0 * TRACK_R2

# The unzipping gate. The molecule does not open all at once: it opens where it
# has stepped back in y, and the track only ever does that once it is out of the
# turn and running along the bottom of the frame. Below ``TRACK_Y_WOUND`` the
# helix is untouched, above ``TRACK_Y_OPEN`` it is split as far as the scene has
# dialled it, and in between it splays - so the fork stands still on screen and
# the molecule unzips itself by flying through it.
#
# Both numbers are read by :class:`DNAModifier` as well as written into the
# track here, for the same reason the scene reads ``marks`` rather than typing
# arc lengths of its own: retuning the path cannot silently leave the gate
# somewhere the molecule never reaches.
TRACK_Y_WOUND = 6.0  # y at which the strands begin to come apart
TRACK_Y_OPEN = 9.0  # y beyond which they are all the way apart
TRACK_OPEN_RUN = 8.0  # how far along the last straight that takes


def _smoothstep(u):
    """3u^2 - 2u^3, clamped. Zero slope at both ends, so two segments joined
    with it meet without a kink - which matters here because a kink in the
    track is a kink in the molecule."""
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    return u * u * (3.0 - 2.0 * u)


def dna_flight_path():
    """The track, as a list of ``(x, y, z)`` sampled every ``TRACK_STEP``.

    Five segments, each picking up where the last left off and with the same
    tangent, so the whole thing is smooth:

    1. straight in from the right, level, in the upper half;
    2. a full 360 degree loop in the middle of the upper half. ``y`` ramps
       across it, so where the path crosses itself on screen it is really
       ``TRACK_DEPTH`` apart in depth and nothing intersects;
    3. on towards the left border, climbing to ``TRACK_Z_MID`` and coming back
       to ``y = 0``;
    4. the turn: 180 degrees, so it comes out heading right again, a full
       ``2 * TRACK_R2`` lower than it went in - which is the whole reason
       segment 3 climbs. The turn has to come out at ``TRACK_Z_OUT``, near the
       middle of the frame, and a hairpin worth looking at has to be a few
       units across, so it must go in that much higher;
    5. straight out to the right, just below the middle, and off the screen.

    ``y`` is doing two jobs at once, and they are worth keeping apart. Across
    the loop it is *depth*: the loop crosses its own entry on screen, and
    stepping back by ``TRACK_DEPTH`` is what keeps the molecule from flying
    through itself. From the turn onwards it is a *signal*: nothing crosses
    anything down there any more, so the last segment's step back to
    ``TRACK_Y_OPEN`` is free to mean "past the fork" - see the constants above.
    Every move of ``y`` is a smoothstep between values that match at the joins,
    so the depth never jumps; a jump in y is not a shortcut but a length of
    track pointing straight at the camera, which the arc-length resampling
    below would faithfully fill with base pairs piled on one spot.

    :return: ``(points, length, marks)`` - the samples, the total arc length,
        and the arc length at which each segment ends, keyed by name. The scene
        times its beats against ``marks`` rather than against numbers of its
        own, so retuning the track above cannot silently desynchronise them.
    """
    pts = [(TRACK_X_IN, 0.0, TRACK_Z_UP)]
    ends = []
    distances = []

    cumulated_length = 0

    def straight(x0, z0, x1, z1, y0, y1,
                 n, y_over=1.0):
        """One straight run, with ``y`` and ``z`` eased across it.

        :param y_over: the fraction of the segment ``y`` makes its move in.
            The default spends the whole segment on it; a smaller value gets
            the move done early and then holds, which is how the last straight
            is over the gate while it is still on screen rather than half a
            frame width off the right edge.
        """
        length = 0
        pt_old = (x0, y0, z0)
        for k in range(1, n + 1):
            u = k / n
            pt = (x0 + (x1 - x0) * u,
                  y0 + (y1 - y0) * _smoothstep(u / y_over),
                  z0 + (z1 - z0) * _smoothstep(u))
            pts.append(pt)
            length += (Vector(pt) - Vector(pt_old)).length
            pt_old = pt
        return length

    tau = 2.0 * pi

    # 1 - fly in from the right
    straight_length = straight(TRACK_X_IN, TRACK_Z_UP, TRACK_X_LOOP, TRACK_Z_UP, 0.0, 0.0, 400)
    ends.append(("loop_start", len(pts) - 1))
    print("loop_start from strand length:", cumulated_length, "to", cumulated_length + straight_length)
    cumulated_length += straight_length
    distances.append(("loop_start", cumulated_length))

    # 2 - the roller-coaster loop
    n = 400
    pt_old = (TRACK_X_LOOP - TRACK_R1 * math.sin(0),
              TRACK_DEPTH * _smoothstep(0 / tau),
              TRACK_Z_UP + TRACK_R1 - TRACK_R1 * math.cos(0))
    loop_length = 0
    for k in range(1, n + 1):
        a = tau * k / n
        pt = (TRACK_X_LOOP - TRACK_R1 * math.sin(a),
              TRACK_DEPTH * _smoothstep(a / tau),
              TRACK_Z_UP + TRACK_R1 - TRACK_R1 * math.cos(a))
        loop_length += (Vector(pt) - Vector(pt_old)).length
        pts.append(pt)
        pt_old = pt

    ends.append(("loop_end", len(pts) - 1))
    print("loop_end from strand length:", cumulated_length, "to", cumulated_length + loop_length)
    cumulated_length += loop_length
    distances.append(("loop_end", cumulated_length))

    # 3 - on to the left, climbing to the height the turn needs to start at,
    # and back out of the loop's depth
    straight_length2 = straight(TRACK_X_LOOP, TRACK_Z_UP, TRACK_X_LEFT, TRACK_Z_MID,
                                TRACK_DEPTH, 0.0, 400)
    ends.append(("turn_start", len(pts) - 1))
    print("turn_start from strand length:", cumulated_length, "to", cumulated_length + straight_length2)
    cumulated_length += straight_length2
    distances.append(("turn_start", cumulated_length))

    # 4 - the 180 degree turn, ending level and heading right. It also spends
    # itself getting y up to the near edge of the gate, so that the molecule
    # comes out of the turn with the fork immediately ahead of it.
    n = 600
    turn_length = 0
    pt_old = (TRACK_X_LEFT - TRACK_R2 * math.sin(0),
              TRACK_Y_WOUND * _smoothstep(0 / (0.5 * tau)),
              TRACK_Z_MID - TRACK_R2 + TRACK_R2 * math.cos(0))
    for k in range(1, n + 1):
        b = 0.5 * tau * k / n  # half a circle
        pt = (TRACK_X_LEFT - TRACK_R2 * math.sin(b),
              TRACK_Y_WOUND * _smoothstep(b / (0.5 * tau)),
              TRACK_Z_MID - TRACK_R2 + TRACK_R2 * math.cos(b))
        turn_length += (Vector(pt) - Vector(pt_old)).length
        pts.append(pt)
        pt_old = pt

    ends.append(("turn_end", len(pts) - 1))
    print("turn_end from strand length:", cumulated_length, "to", cumulated_length + turn_length)
    cumulated_length += turn_length
    distances.append(("turn_end", cumulated_length))

    # 5 - away to the right and off the screen, crossing the gate as it goes
    # and staying over it: anything that drifted back under TRACK_Y_WOUND here
    # would wind itself up again on the way out.
    z_out = TRACK_Z_OUT
    length3 = straight(TRACK_X_LEFT, z_out, TRACK_X_OUT, z_out,
                       TRACK_Y_WOUND, TRACK_Y_OPEN, 900,
                       y_over=TRACK_OPEN_RUN / (TRACK_X_OUT - TRACK_X_LEFT))

    print("end_point from strand length:", cumulated_length, "to", cumulated_length + length3)
    cumulated_length += length3
    distances.append(("end_point", cumulated_length))

    # resample at a uniform arc length, so that "length along the track" and
    # "distance travelled" are the same number - the molecule is spaced by arc
    # length and would otherwise bunch up wherever the samples happen to crowd
    cumulative = [0.0]
    for a, b in zip(pts, pts[1:]):
        step = math.sqrt(sum((q - p) ** 2 for p, q in zip(a, b)))
        cumulative.append(cumulative[-1] + step)
    total = cumulative[-1]

    even, j = [], 0
    s = 0.0
    while s < total:
        while j < len(cumulative) - 2 and cumulative[j + 1] < s:
            j += 1
        span = cumulative[j + 1] - cumulative[j]
        f = 0.0 if span == 0 else (s - cumulative[j]) / span
        even.append(tuple(p + (q - p) * f for p, q in zip(pts[j], pts[j + 1])))
        s += TRACK_STEP
    return even, total, {name: cumulative[i] for name, i in ends}, {name: length for name, length in distances}


def write_dna_track(path):
    """Put the track where an ``Import CSV`` node can read it.

    The first line is spent on the column header - ``Import CSV`` always does
    that and has no option not to - so the columns arrive in the graph as three
    float attributes called X, Y and Z.

    :return: ``(length, marks)`` - see :func:`dna_flight_path`.
    """
    points, total, marks, distances = dna_flight_path()
    with open(path, "w") as file:
        file.write("X,Y,Z\n")
        for x, y, z in points:
            file.write("%.5f,%.5f,%.5f\n" % (x, y, z))
    return total, marks, distances


class MorphModifier(GeometryNodesModifier):
    """One shape's points slid onto another's - a picture frame into an arrow.

    A port of ``video_bff/tmp.xml``, the tree authored in the Blender editor.
    Three frames of nodes, the same three the editor shows:

    ``Object 1``
        the shape morphed *from*: a rectangle swept along a small circle
        (``Curve to Mesh``), which makes a tube bent into a picture frame,
        stood upright and moved off to the left.
    ``Object 2``
        the shape morphed *to*: a cone on top of a cylinder that has been
        dropped by half its length, joined into an arrow pointing up.
    ``Morphing``
        the blend itself: the straight line between the two shapes,
        ``(1 - t) * here + t * there``, with ``t`` the ``MorphParameter``
        value node. Animating that from 0 to 1 in the scene is the whole
        animation.

    The two shape frames are the editor's tree node for node; the third is
    the one deliberate difference. What the editor draws as six loose nodes
    (``Position``, ``Index``, ``Sample Index``, two scales, an add and a
    ``Set Position``) is one :class:`~geometry_nodes.nodes.MorphNode` group
    here, because that blend is not specific to frames and arrows and reads
    better as a node than as a diagram. The geometry it produces is identical
    - the group holds exactly those nodes - so the only thing lost is the
    view of them from outside, and a click on the group brings that back.

    Two things follow from pairing the two shapes by index, which is what
    ``match_nearest=False`` asks :class:`~geometry_nodes.nodes.MorphNode` for
    and what the editor's tree did:

    - the point that ends up at the arrow's tip is whichever point of the
      frame happens to share the tip's index, so the frame turns itself
      inside out on the way rather than folding neatly;
    - the two shapes do not have the same number of points: the frame's tube
      carries 128, the arrow 100 (see :meth:`point_counts`). ``MorphNode``
      rescales the index so the mismatch costs resolution rather than
      correctness - several frame points share an arrow point - but it cannot
      invent the points that would make the pairing a bijection. Matching the
      counts exactly (``frame_resolution=25``) tightens the result.

    :param frame_width: width of the rectangle.
    :param frame_height: height of the rectangle.
    :param frame_thickness: radius of the circle swept along it.
    :param frame_resolution: segments of that circle.
    :param frame_location: where the frame sits before it morphs.
    :param arrow_head_radius: radius of the cone's base.
    :param arrow_head_length: length of the cone.
    :param arrow_shaft_radius: radius of the cylinder.
    :param arrow_shaft_length: length of the cylinder.
    :param arrow_resolution: vertices around both of them.
    :param morph: starting value of ``MorphParameter``.
    :param match_nearest: which pairing :class:`~geometry_nodes.nodes.MorphNode`
        uses - nearest point on the target surface, or the rescaled index.
        The two differ sharply on this pair of shapes; see the scene.
    :param color: palette name for the morphing shape, or ``None`` for the
        tree exactly as the editor has it. The xml carries no ``Set
        Material`` node, and a material sitting in the *object's* slot does
        not reach geometry that nodes create - the evaluated mesh brings its
        own (empty) material list - so a shape built entirely out of
        primitives like this one renders in blender's default grey until the
        graph itself paints it. Naming a colour here adds that one node.
    """

    def __init__(self, frame_width=2.0, frame_height=2.0, frame_thickness=0.1,
                 frame_resolution=32, frame_location=(-5.3, 0, 0),
                 arrow_head_radius=0.5, arrow_head_length=1.0,
                 arrow_shaft_radius=0.15, arrow_shaft_length=1.0,
                 arrow_resolution=32, morph=0.0, color=None, match_nearest=True,
                 name="Morph", **kwargs):
        self.color = color
        self.match_nearest = match_nearest
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_thickness = frame_thickness
        self.frame_resolution = frame_resolution
        self.frame_location = Vector(frame_location)
        self.arrow_head_radius = arrow_head_radius
        self.arrow_head_length = arrow_head_length
        self.arrow_shaft_radius = arrow_shaft_radius
        self.arrow_shaft_length = arrow_shaft_length
        self.arrow_resolution = arrow_resolution
        self.morph = morph
        self.kwargs = kwargs
        super().__init__(name=name, automatic_layout=False)

    # ----------------------------------------------------------------
    def point_counts(self):
        """``(frame, arrow)`` point counts, which the morph maps index to index.

        Worked out the way blender builds the two, so that the mismatch the
        class docstring warns about can be seen without opening the editor:
        a closed curve of four corners swept along a circle of ``n``
        segments carries ``4 * n`` points, while the arrow is a cone (a ring,
        a tip and the centre of its fan-filled base) plus a cylinder (two
        rings and two fan centres).
        """
        frame = 4 * self.frame_resolution
        cone = self.arrow_resolution + 2
        cylinder = 2 * self.arrow_resolution + 2
        return frame, cone + cylinder

    # ----------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        source = self._create_frame_frame(tree)
        target = self._create_arrow_frame(tree)
        morphed = self._create_morphing_frame(tree, source, target)

        joined = JoinGeometry(tree, location=(11, -1.2))
        tree.links.new(morphed, joined.geometry_in)

        out = self.group_outputs.inputs["Geometry"]
        if self.color is None:
            tree.links.new(joined.geometry_out, out)
        else:
            material = InputMaterial(tree, location=(11.6, -1.9), material=self.color,
                                     name="MorphColor", **self.kwargs, hide=True)
            self.materials.append(material.node.material)
            painted = SetMaterial(tree, location=(11.8, -1.2),
                                  material=material.std_out, name="PaintMorph")
            create_geometry_line(tree, [joined, painted])
            tree.links.new(painted.geometry_out, out)

    # ----------------------------------------------------------------
    def _create_frame_frame(self, tree):
        """``Object 1``: a rectangle swept along a circle, stood upright.

        :return: the geometry socket of the frame, where it starts out.
        """
        rectangle = Quadrilateral(tree, location=(0.1, -0.3), mode="RECTANGLE",
                                  width=self.frame_width, height=self.frame_height,
                                  name="FrameRectangle")
        profile = CurveCircle(tree, location=(0.1, -0.9), mode="RADIUS",
                              resolution=self.frame_resolution,
                              radius=self.frame_thickness, name="FrameProfile")
        tube = CurveToMesh(tree, location=(1.0, -0.1), curve=rectangle.geometry_out,
                           profile_curve=profile.geometry_out, fill_caps=False,
                           name="FrameTube")
        # upright and off to the left: the rectangle is built lying in x-y and
        # the shot looks along +y, so without the quarter turn it would be
        # seen edge on
        placed = TransformGeometry(tree, location=(1.9, -0.3),
                                   translation=self.frame_location,
                                   rotation=[pi / 2, 0, 0], name="PlaceFrame")
        tree.links.new(tube.geometry_out, placed.geometry_in)

        node_frame = Frame(tree, location=(-2.1, 0.9), label="Object 1")
        node_frame.add([rectangle, profile, tube, placed])
        return placed.geometry_out

    # ----------------------------------------------------------------
    def _create_arrow_frame(self, tree):
        """``Object 2``: a cone sitting on a cylinder, joined into an arrow.

        The cylinder is dropped by half its length so that its top meets the
        cone's base at the origin; the cone is left where it is built, which
        is why the arrow points up out of the origin rather than being
        centred on it.

        :return: the geometry socket of the arrow.
        """
        head = ConeMesh(tree, location=(0.1, -0.1), vertices=self.arrow_resolution,
                        radius_top=0, radius_bottom=self.arrow_head_radius,
                        depth=self.arrow_head_length, name="ArrowHead")
        shaft = CylinderMesh(tree, location=(0.1, -1.7), vertices=self.arrow_resolution,
                             radius=self.arrow_shaft_radius,
                             depth=self.arrow_shaft_length, name="ArrowShaft")
        dropped = TransformGeometry(tree, location=(1.0, -1.2),
                                    translation=[0, 0, -self.arrow_shaft_length / 2],
                                    name="DropShaft")
        tree.links.new(shaft.geometry_out, dropped.geometry_in)

        # head first, then shaft: the join fixes the point order the morph
        # then reads by index, so swapping these swaps which part of the
        # frame ends up as the tip
        arrow = JoinGeometry(tree, location=(2.2, -0.8), name="Arrow")
        tree.links.new(head.geometry_out, arrow.geometry_in)
        tree.links.new(dropped.geometry_out, arrow.geometry_in)

        node_frame = Frame(tree, location=(-2.1, -1.1), label="Object 2")
        node_frame.add([head, shaft, dropped, arrow])
        return arrow.geometry_out

    # ----------------------------------------------------------------
    def _create_morphing_frame(self, tree, source, target):
        """``Morphing``: walk every point of ``source`` towards ``target``.

        The six nodes the editor's ``Morphing`` frame holds live in
        :class:`MorphNode` now, so what is left here is the value that drives
        them. It is still a ``Value`` node called ``MorphParameter`` rather
        than the group's own socket default, because that is what the scene
        keyframes - a socket on a group node can be animated too, but the
        ``Value`` node is what ``ibpy.get_geometry_node_from_modifier`` finds
        by name.

        :param source: geometry whose points are moved.
        :param target: geometry sampled, by index, for where to move them to.
        :return: the geometry socket of the morphed source.
        """
        parameter = InputValue(tree, location=(3.0, -2.6), value=self.morph,
                               name="MorphParameter")
        morph = MorphNode(tree, location=(3.8, -0.2), geometry1=source,
                          geometry2=target, morph_parameter=parameter.std_out,
                          match_nearest=self.match_nearest, name="Morph")

        node_frame = Frame(tree, location=(4.5, -0.5), label="Morphing")
        node_frame.add([parameter, morph])
        return morph.geometry_out


class OutlineMorphModifier(GeometryNodesModifier):
    """The frame and the arrow again, this time as shapes that can morph.

    :class:`MorphModifier` is the editor's tree ported faithfully, and it is
    the reason ``docs/theory_morphing.tex`` exists: a tube bent into a
    rectangle and a solid cone on a cylinder have neither the same number of
    points nor the same topology, so no correspondence between them is any
    good, and both of ``MorphNode``'s pairings can only pick which way it
    looks wrong.

    The theory's own first answer is not a better pairing but *compatible
    shapes*, and for outlines it costs nothing: both are built as closed
    curves, both are resampled to the same number of points, and both are
    swept along the same profile circle. That makes the two meshes the same
    mesh twice over - equal counts, equal connectivity, and index *i* the
    same fraction of the way around each outline - so plain index pairing is
    exactly right and the morph is a clean interpolation with no folding,
    no knot, and no shrink-wrap.

    The arrow is its own silhouette here: the outline a cone standing on a
    cylinder actually shows to a camera looking at it side on. That is the
    price - it is a flat tube like the frame, not a solid - and it is what
    makes the two shapes the same kind of thing, which is the whole point.
    For two *solids*, the equivalent move is the sphere remeshing of
    ``geometry_nodes/supermesh.py``.

    ``Resample Curve`` samples by arc length, so the correspondence is
    geometric rather than an accident of storage: point *i* sits at the same
    fraction of the way around each outline. Both curves also have to run
    the same way round, or the loop would turn inside out on the way; the
    arrow's outline is written to match the rectangle's winding.

    :param samples: points each outline is resampled to. Corners are cut by
        up to half the sample spacing, so this is a sharpness dial.
    :param profile_resolution: segments of the circle both outlines are
        swept along.
    :param thickness: radius of that circle.
    :param frame_width: width of the rectangle.
    :param frame_height: height of the rectangle.
    :param frame_location: where the frame waits before it morphs.
    :param head_radius: half the width of the arrow's barbs.
    :param head_length: length of its head.
    :param shaft_radius: half the width of its shaft.
    :param shaft_length: how far the shaft reaches below the head's base.
    :param morph: starting value of ``MorphParameter``.
    :param color: palette name, or ``None`` to leave the geometry unpainted.
    """

    def __init__(self, samples=128, profile_resolution=32, thickness=0.1,
                 frame_width=2.0, frame_height=2.0, frame_location=(-5.3, 0, 0),
                 head_radius=0.5, head_length=1.0, shaft_radius=0.15,
                 shaft_length=1.0, morph=0.0, color=None,
                 name="OutlineMorph", **kwargs):
        self.samples = samples
        self.profile_resolution = profile_resolution
        self.thickness = thickness
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_location = Vector(frame_location)
        self.head_radius = head_radius
        self.head_length = head_length
        self.shaft_radius = shaft_radius
        self.shaft_length = shaft_length
        self.morph = morph
        self.color = color
        self.kwargs = kwargs
        super().__init__(name=name, automatic_layout=False)

    # ----------------------------------------------------------------
    @property
    def arrow_outline(self):
        """The arrow's silhouette as a closed polygon, in the build plane.

        Seven corners, starting at the tip and running down the left barb,
        round the foot of the shaft and back up the right barb. Both the
        starting corner and the direction are chosen, not arbitrary:

        - ``Resample Curve`` numbers its samples from the curve's first
          control point, so the corner named first here is the one the
          rectangle's own first corner - its top right - will travel to.
          Starting at the tip is what makes the morph read as a corner
          reaching out and becoming the point of the arrow.
        - The winding has to match the rectangle's, which runs the positive
          way round. Reverse this list and every point would be paired with
          one on the far side of the loop, so the tube would turn itself
          inside out on the way across - the signed area of the two outlines
          is the thing to check if that ever looks wrong.

        The numbers are the ones the solid arrow of :class:`MorphModifier` is
        built from, so the two are the same arrow seen two ways: the cone
        spans ``+-head_length/2`` and the shaft hangs from the cone's base
        down to ``-shaft_length``.
        """
        r, R = self.shaft_radius, self.head_radius
        tip, base = self.head_length / 2, -self.head_length / 2
        foot = -self.shaft_length
        return [(0, tip), (-R, base), (-r, base), (-r, foot),
                (r, foot), (r, base), (R, base)]

    @property
    def point_count(self):
        """Points in either swept outline - they agree, which is the point."""
        return self.samples * self.profile_resolution

    # ----------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        profile = CurveCircle(tree, location=(0, -8), mode="RADIUS",
                              resolution=self.profile_resolution,
                              radius=self.thickness, name="SweepProfile")
        frame = Frame(tree, label="Profile")
        frame.add([profile])

        source = self._create_frame_frame(tree, profile)
        target = self._create_arrow_frame(tree, profile)
        morphed = self._create_morphing_frame(tree, source, target)

        out = self.group_outputs.inputs["Geometry"]
        if self.color is None:
            tree.links.new(morphed, out)
        else:
            material = InputMaterial(tree, location=(17, -3), material=self.color,
                                     name="MorphColor", **self.kwargs, hide=True)
            self.materials.append(material.node.material)
            painted = SetMaterial(tree, location=(17, -1.4),
                                  material=material.std_out, name="PaintMorph")
            tree.links.new(morphed, painted.geometry_in)
            tree.links.new(painted.geometry_out, out)

    # ----------------------------------------------------------------
    def _sweep(self, tree, curve, profile, x, label):
        """Resample a closed outline to ``samples`` points and sweep it.

        The two calls to this are what make the shapes compatible: same
        count, same profile, same order of operations, so the two meshes
        come out with the same connectivity and the same point count.
        """
        resampled = ResampleCurve(tree, location=(x, -2), mode="Count",
                                  curve=curve, count=self.samples,
                                  name=label + "Samples")
        tube = CurveToMesh(tree, location=(x + 1.2, -2), curve=resampled.geometry_out,
                           profile_curve=profile.geometry_out, fill_caps=False,
                           name=label + "Tube")
        return resampled, tube

    # ----------------------------------------------------------------
    def _create_frame_frame(self, tree, profile):
        """``Outline 1``: the rectangle, resampled and swept."""
        rectangle = Quadrilateral(tree, location=(0, -2), mode="RECTANGLE",
                                  width=self.frame_width, height=self.frame_height,
                                  name="FrameRectangle")
        resampled, tube = self._sweep(tree, rectangle.geometry_out, profile,
                                      1.4, "Frame")
        placed = TransformGeometry(tree, location=(3.8, -2),
                                   geometry=tube.geometry_out,
                                   translation=self.frame_location,
                                   rotation=[pi / 2, 0, 0], name="PlaceFrame")

        node_frame = Frame(tree, label="Outline 1")
        node_frame.add([rectangle, resampled, tube, placed])
        return placed.geometry_out

    # ----------------------------------------------------------------
    def _create_arrow_frame(self, tree, profile):
        """``Outline 2``: the arrow's silhouette, resampled and swept.

        A point cloud of seven corners, ordered by an ``Index Switch``, joined
        into one curve and closed. Blender has no "polygon from a list of
        points" node, and this is the shortest way to say it that keeps the
        corner coordinates visible in the editor.
        """
        corners = [InputVector(tree, location=(0, -4.4 - 0.3 * i),
                               value=Vector((x, y, 0)), name="Corner%d" % i,
                               hide=True)
                   for i, (x, y) in enumerate(self.arrow_outline)]
        index = Index(tree, location=(0, -4.1), hide=True)
        pick = IndexSwitch(tree, location=(1.4, -4.4), data_type="VECTOR",
                           index=index.std_out, name="CornerAt")
        for corner in corners:
            pick.add_item(socket=corner.std_out)

        points = Points(tree, location=(1.4, -3.4), count=len(corners),
                        name="ArrowCorners")
        placed_points = SetPosition(tree, location=(2.6, -3.4),
                                    geometry=points.geometry_out,
                                    position=pick.std_out, name="PlaceCorners")
        polygon = PointsToCurve(tree, location=(3.6, -3.4), name="ArrowOutline")
        tree.links.new(placed_points.geometry_out, polygon.geometry_in)
        closed = SetSplineCyclic(tree, location=(4.6, -3.4), cyclic=True,
                                 name="CloseArrow")
        tree.links.new(polygon.geometry_out, closed.geometry_in)

        resampled, tube = self._sweep(tree, closed.geometry_out, profile,
                                      5.8, "Arrow")
        # the arrow is built lying in the same plane as the rectangle and
        # stood up the same way, so the two shapes differ only in outline
        placed = TransformGeometry(tree, location=(8.2, -2),
                                   geometry=tube.geometry_out,
                                   rotation=[pi / 2, 0, 0], name="PlaceArrow")

        node_frame = Frame(tree, label="Outline 2")
        node_frame.add(corners + [index, pick, points, placed_points, polygon,
                                  closed, resampled, tube, placed])
        return placed.geometry_out

    # ----------------------------------------------------------------
    def _create_morphing_frame(self, tree, source, target):
        """``Morphing``: index pairing, which is exact for compatible shapes."""
        parameter = InputValue(tree, location=(12, -3.4), value=self.morph,
                               name="MorphParameter")
        morph = MorphNode(tree, location=(13.4, -1.4), geometry1=source,
                          geometry2=target, morph_parameter=parameter.std_out,
                          match_nearest=False, name="Morph")

        node_frame = Frame(tree, label="Morphing")
        node_frame.add([parameter, morph])
        return morph.geometry_out


class TubeMorphModifier(GeometryNodesModifier):
    """The frame unrolled into an arrow, by way of :class:`MorphNode2`.

    A third answer to the same problem, and the one that gives a *solid*
    arrow back. Both shapes are written as a curve carrying a radius:

    ``Curve 1``
        the frame - a rectangle of constant radius, which swept is the tube
        it always was.
    ``Curve 2``
        the arrow's axis - a straight segment from the foot of the shaft to
        the tip, whose radius holds at ``shaft_radius``, jumps out to
        ``head_radius`` where the barbs are and falls to zero at the point.
        Swept, that is a cylinder with a cone on it: an arrow is a solid of
        revolution, so a curve and a radius is all it takes to say so.

    :class:`MorphNode2` then blends the two paths and the two radius
    profiles at once, so the loop straightens while the thickness grows. The
    frame is cut open to do it - a loop has two ends once you cut it, an axis
    has two ends already, and that is what makes the two the same kind of
    object. The gap this leaves is one sample spacing wide (``8/samples`` of
    a unit here) and shows as a notch in the frame before the morph starts;
    ``close_loop=True`` sweeps it shut at the price of a bridge across the
    arrow at the other end.

    Compared with the two other modifiers over the same pair of shapes:
    :class:`MorphModifier` is the editor's tree, which has no correspondence
    worth the name; :class:`OutlineMorphModifier` makes both shapes flat
    outlines, which morphs cleanly but cannot be solid; this one keeps the
    arrow solid, at the price of the cut and of a mid-morph shape that is a
    thickening curl rather than anything you could name.

    :param samples: points along both curves - the arrow's radius profile is
        built at this resolution too, so it also sets how sharp the barbs are.
    :param profile_resolution: segments of the swept circle.
    :param thickness: the frame's tube radius.
    :param close_loop: sweep the blend closed instead of cutting it open.
    """

    def __init__(self, samples=128, profile_resolution=16, thickness=0.1,
                 frame_width=2.0, frame_height=2.0, frame_location=(-5.3, 0, 0),
                 head_radius=0.5, head_length=1.0, shaft_radius=0.15,
                 shaft_length=1.0, morph=0.0, color=None, close_loop=False,
                 name="TubeMorph", **kwargs):
        self.samples = samples
        self.profile_resolution = profile_resolution
        self.thickness = thickness
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_location = Vector(frame_location)
        self.head_radius = head_radius
        self.head_length = head_length
        self.shaft_radius = shaft_radius
        self.shaft_length = shaft_length
        self.morph = morph
        self.color = color
        self.close_loop = close_loop
        self.kwargs = kwargs
        super().__init__(name=name, automatic_layout=False)

    # ----------------------------------------------------------------
    @property
    def axis_ends(self):
        """``(foot, tip)`` of the arrow's axis, matching the solid arrow.

        The cone spans ``+-head_length/2`` and the shaft hangs from its base
        down to ``-shaft_length``, which is where :class:`MorphModifier`
        leaves them.
        """
        return -self.shaft_length, self.head_length / 2

    @property
    def barb_fraction(self):
        """Where along the axis the shaft stops and the head begins, in 0..1."""
        foot, tip = self.axis_ends
        return (-self.head_length / 2 - foot) / (tip - foot)

    # ----------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        source = self._create_frame_frame(tree)
        target = self._create_axis_frame(tree)

        parameter = InputValue(tree, location=(8, -6), value=self.morph,
                               name="MorphParameter")
        morph = MorphNode2(tree, location=(9.4, -2), curve1=source, curve2=target,
                           morph_parameter=parameter.std_out, samples=self.samples,
                           profile_resolution=self.profile_resolution,
                           close_loop=self.close_loop, name="Morph")
        node_frame = Frame(tree, label="Morphing")
        node_frame.add([parameter, morph])

        out = self.group_outputs.inputs["Geometry"]
        if self.color is None:
            tree.links.new(morph.geometry_out, out)
        else:
            material = InputMaterial(tree, location=(13, -4), material=self.color,
                                     name="MorphColor", **self.kwargs, hide=True)
            self.materials.append(material.node.material)
            painted = SetMaterial(tree, location=(13, -2),
                                  material=material.std_out, name="PaintMorph")
            tree.links.new(morph.geometry_out, painted.geometry_in)
            tree.links.new(painted.geometry_out, out)

    # ----------------------------------------------------------------
    def _create_frame_frame(self, tree):
        """``Curve 1``: the rectangle, at constant radius, stood upright."""
        rectangle = Quadrilateral(tree, location=(0, -1), mode="RECTANGLE",
                                  width=self.frame_width, height=self.frame_height,
                                  name="FrameRectangle")
        thick = SetCurveRadius(tree, location=(1.4, -1), curve=rectangle.geometry_out,
                               radius=self.thickness, name="FrameThickness")
        placed = TransformGeometry(tree, location=(2.8, -1),
                                   geometry=thick.geometry_out,
                                   translation=self.frame_location,
                                   rotation=[pi / 2, 0, 0], name="PlaceFrame")

        node_frame = Frame(tree, label="Curve 1 - the frame")
        node_frame.add([rectangle, thick, placed])
        return placed.geometry_out

    # ----------------------------------------------------------------
    def _create_axis_frame(self, tree):
        """``Curve 2``: the arrow's axis, with the arrow written as a radius.

        The line is resampled *before* the radius is written, because a
        profile is only as detailed as the points that carry it: set on the
        two ends of a bare segment, it could only ever be a straight taper.
        """
        foot, tip = self.axis_ends
        line = CurveLine(tree, location=(0, -4), start=Vector((0, 0, foot)),
                         end=Vector((0, 0, tip)), name="ArrowAxis")
        sampled = ResampleCurve(tree, location=(1.4, -4), mode="Count",
                                curve=line.geometry_out, count=self.samples,
                                name="AxisSamples")

        factor = SplineParameter(tree, location=(1.4, -5.4), std_out="Factor",
                                 name="AlongAxis")
        profile = make_function(
            tree, name="ArrowRadius", location=(2.8, -5.4), hide=True,
            aux_functions={
                # 1 while the shaft lasts, 0 once the head starts
                "shaft": "u,%.6f,<" % self.barb_fraction,
                # the head is a straight taper from the barbs to nothing
                "head": "1,u,-,%.6f,/,%.6f,*" % (1 - self.barb_fraction,
                                                 self.head_radius),
            },
            functions={"radius": "shaft,%.6f,*,1,shaft,-,head,*,+" % self.shaft_radius},
            inputs=["u"], outputs=["radius"],
            scalars=["u", "shaft", "head", "radius"])
        tree.links.new(factor.std_out, profile.inputs["u"])

        shaped = SetCurveRadius(tree, location=(4.2, -4), curve=sampled.geometry_out,
                                radius=profile.outputs["radius"], name="ArrowProfile")

        node_frame = Frame(tree, label="Curve 2 - the arrow")
        node_frame.add([line, sampled, factor, profile, shaped])
        return shaped.geometry_out


class DNAModifier(GeometryNodesModifier):
    """A DNA double helix that flies along a track.

    Rebuilt from ``video_bff/tmp.xml``, the node tree that was authored in the
    Blender editor, with the sine that used to bend the axis replaced by
    :func:`dna_flight_path` - see the comment above it for why.

    The molecule is built in seven steps, and each is a frame in the editor:

    ``ControlFrame``
        Every number the helix depends on, in one column, plus the palette. The
        five that the scene animates - ``HeadOffset``, ``StrandSeparation``,
        ``TiltLength``, ``BaseSize`` and ``PeelHeight`` - are the whole of the
        choreography. Three of them have a ``Default*`` twin that is never
        keyframed; see ``Unzipping Gate``.
    ``Flight Path``
        The track, read from csv, and the arc length each point of the molecule
        sits at: ``HeadOffset - Index * Spacing``.
    ``LinearStructure``
        A line of ``pairs`` points, moved onto the track.
    ``Unzipping Gate``
        One number per point, from its own ``y``: 0 where the molecule is still
        a double helix, 1 where it has come apart, smoothstepped between
        ``TRACK_Y_WOUND`` and ``TRACK_Y_OPEN`` in between. The molecule does not
        open all at once - it opens *where it is*, and the track only steps back
        in ``y`` once it is out of the turn, so the fork stands still at the
        bottom left of the frame and the molecule unzips itself by flying
        through it.
    ``Helix``
        The line is tilted by an amount that grows with the index, duplicated
        into two splines, and the two are pushed apart along the curve normal.
        Because the tilt turns the normal as it goes, "pushed apart along the
        normal" is what makes the pair of strands wind around each other - and
        because the tilt is driven by the *index* rather than by arc length,
        the helix is fixed in the molecule and the molecule slides along the
        track without appearing to screw itself forward. All of it is built
        twice, wound and open, and the gate mixes the two.
    ``Strand Shift``
        The two strands slide along the tangent in opposite directions, which
        opens the major and minor groove. This frame also holds ``PeelHeight``:
        where the helix has unwound, lifting one strand takes it off the top of
        the frame and leaves the other behind.
    ``Base Pair Coloring``
        Each point is given a ``BaseType`` in 0..3, so that the two strands of
        a pair always carry complementary bases.
    ``Bases And Pairing``
        A for-each zone builds one base per point, aimed at its partner.
    ``Strand Geometry``
        The two backbones, swept to tubes, with a sphere on every point.

    Three deliberate departures from the xml:

    * The three ``Switch`` nodes that chose an RGBA colour and the
      ``Store Named Attribute`` that wrote it to ``C`` are gone. The same three
      switches are now ``input_type='MATERIAL'`` and pick one of the project
      colours ``custom1 / joker / important / drawing``. A material is not a
      field, so they cannot live where the colour switches did: they sit inside
      the for-each zone, where one iteration means one base.
    * A ``Bit Math`` node hung unconnected in the ``Helix`` frame of the xml. It
      fed nothing and is not reproduced.
    * The xml's ``Selective Untwisting`` and ``Selective StrandOpening`` frames
      each compared ``y`` against 6 and switched a *parameter*: ``TiltLength``
      or ``StrandSeparation``, default on one side of the line and dialled on
      the other. Those two frames are one ``Unzipping Gate`` here, and what it
      feeds is not a parameter but a crossfade between two whole helices. The
      reason is in ``_create_helix_frame``: a Tilt Length that varies from
      point to point does not describe a molecule that is unwinding, it
      describes one that is being wrung out, and the shorter the fork the
      more violently. Switching sharply, as the xml does, hides that in a
      single bad segment; smoothing it - which is what this is for - would
      have spread it over the whole fork.

    :param pairs: number of base pairs. Each contributes one point per strand.
    :param spacing: distance between base pairs along the track. The molecule
        is therefore ``(pairs - 1) * spacing`` long.
    :param head_offset: where the leading end starts out along the track.
        Animate this and the molecule flies.
    :param tilt_length: index period of the helical twist. Larger unwinds it.
    :param strand_separation: distance between the two backbones. Larger splits
        them apart.
    :param strand_shift: how far the two strands slide apart along the tangent -
        the width of the minor groove.
    :param base_size: scale of one base. Zero makes the bases vanish.
    :param peel_height: how far the second strand is lifted in world z.
    :param base_colors: the four base materials, in ``BaseType`` order.
    :param strand_color: material of the two swept backbones.
    :param molecule_color: material of the spheres sitting on the backbones.
    """

    # radii of the swept tubes and of the spheres, as the xml had them
    BACKBONE_RADIUS = 0.10
    BACKBONE_SPHERE_RADIUS = 0.13
    BASE_BOND_RADIUS = 0.10
    BASE_ATOM_RADIUS = 0.17

    # how many atoms a base of each ``BaseType`` is drawn with. Two purines and
    # two pyrimidines, so that a pair is always one long and one short.
    BASE_ATOMS = (4, 5, 2, 3)

    #: the palette of appearance.textures, which the BaseMixing material mixes
    #: too - one place for the colours of the molecule
    BASE_COLORS = DNA_BASE_COLORS

    def __init__(self, pairs=200, spacing=0.34, head_offset=0.0,
                 tilt_length=1.6899462938308716, strand_separation=1.5,
                 strand_shift=0.35, base_size=0.7, peel_height=0.0,
                 base_colors=None, strand_color="gray_4",
                 molecule_color="gray_7", seed=4, name="DNA", **kwargs):
        self.pairs = pairs
        self.spacing = spacing
        self.head_offset = head_offset
        self.tilt_length = tilt_length
        self.strand_separation = strand_separation
        self.strand_shift = strand_shift
        self.base_size = base_size
        self.peel_height = peel_height
        self.base_colors = tuple(base_colors or self.BASE_COLORS)
        self.strand_color = strand_color
        self.molecule_color = molecule_color
        self.seed = seed
        self.kwargs = kwargs

        # the track is data, so it goes in the data directory next to the tapes
        self.track_path = os.path.join(DATA_DIR, DNA_TRACK_FILE + ".csv")
        #: total arc length of the track, and the arc length at which each of
        #: its segments ends - what the scene needs in order to say "start the
        #: split once the turn is behind us" without knowing how the track is
        #: put together
        self.track_length, self.track_marks, self.distances = write_dna_track(self.track_path)
        #: length of the molecule itself, in the same units
        self.molecule_length = (pairs - 1) * spacing

        super().__init__(name=name, automatic_layout=False)

    # ------------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        control = self._create_control_frame(tree)
        flight = self._create_flight_path_frame(tree, control)
        line = self._create_linear_structure_frame(tree, control, flight)
        gate = self._create_unzipping_gate_frame(tree)
        helix = self._create_helix_frame(tree, control, line, gate)
        strands = self._create_strand_shift_frame(tree, control, helix)
        colored = self._create_base_pair_coloring_frame(tree, control, helix,
                                                        strands)
        bases = self._create_bases_and_pairing_frame(tree, control, helix,
                                                     strands, colored)
        backbone = self._create_strand_geometry_frame(tree, control, strands)

        join = JoinGeometry(tree, location=(82.0, 0.0), node_height=GRID,
                            name="JoinMolecule")
        for piece in (backbone, bases):
            tree.links.new(piece, join.geometry_in)
        tree.links.new(join.geometry_out, self.group_outputs.inputs["Geometry"])
        self.group_outputs.location = (83.5 * GRID, 0)

    # ------------------------------------------------------------------
    def _create_control_frame(self, tree):
        """``ControlFrame``: every constant the molecule depends on.

        One column, so that the whole shape of the thing can be changed without
        hunting through the graph for the node that holds the number. The scene
        reaches four of these by label and keyframes them.

        :return: ``{name: node}``, keyed by the label the node carries in the
            editor.
        """
        x = 0.0
        control = {
            "Pairs": InputInteger(tree, location=(x, 0.0), integer=self.pairs,
                                  name="Pairs", node_height=GRID),
            "Spacing": InputValue(tree, location=(x, -0.5), value=self.spacing,
                                  name="Spacing", node_height=GRID),
            "HeadOffset": InputValue(tree, location=(x, -1.0),
                                     value=self.head_offset, name="HeadOffset",
                                     node_height=GRID),
            "TiltLength": InputValue(tree, location=(x, -1.5),
                                     value=self.tilt_length, name="TiltLength",
                                     node_height=GRID),
            "StrandSeparation": InputValue(tree, location=(x, -2.0),
                                           value=self.strand_separation,
                                           name="StrandSeparation",
                                           node_height=GRID),
            "StrandShift": InputValue(tree, location=(x, -2.5),
                                      value=self.strand_shift,
                                      name="StrandShift", node_height=GRID),
            "BaseSize": InputValue(tree, location=(x, -3.0),
                                   value=self.base_size, name="BaseSize",
                                   node_height=GRID),
            "PeelHeight": InputValue(tree, location=(x, -3.5),
                                     value=self.peel_height, name="PeelHeight",
                                     node_height=GRID),
        }

        # The three the scene keyframes to open the molecule have a twin here
        # that it never touches. The twin is what the molecule looks like where
        # it is still wound, the original what it looks like where it is open,
        # and the gate crossfades between them - so before the scene moves a
        # dial the two are equal and the gate has nothing to do, which is why
        # the first seventeen seconds need no keyframes of their own.
        for row, (twin, value) in enumerate([
            ("DefaultTiltLength", self.tilt_length),
            ("DefaultStrandSeparation", self.strand_separation),
            ("DefaultBaseSize", self.base_size)]):
            control[twin] = InputValue(tree, location=(x - 1.5, -1.5 - 0.5 * row),
                                       value=value, name=twin, node_height=GRID)

        # the palette. The four bases first, in BaseType order, then the two
        # colours the backbone is drawn in.
        palette = [("Base%d" % i, color)
                   for i, color in enumerate(self.base_colors)]
        palette += [("Strand", self.strand_color),
                    ("Molecule", self.molecule_color)]
        for row, (node_name, color) in enumerate(palette):
            control[node_name] = InputMaterial(
                tree, location=(x, -4.5 - 0.5 * row), material=color,
                name=node_name, node_height=GRID, **self.kwargs)
        for node_name, _ in palette:
            self.materials.append(control[node_name].node.material)

        # The six materials travel as one bundle rather than as six wires. Two
        # frames want them - the base switches inside the for-each zone and the
        # two backbone Set Material nodes - and both are the whole width of the
        # graph away from here, so what crossed it as six long wires now
        # crosses as one, and the frame that receives it says what it is
        # unpacking in a single Separate Bundle. The Input Material nodes stay
        # where they are: the bundle gathers them rather than replacing them,
        # so each colour is still a node of its own to reach for by name.
        control["Palette"] = CombineBundle(
            tree, location=(1.0, -5.5), name="Palette", node_height=GRID,
            items=[(node_name, "MATERIAL", control[node_name].std_out)
                   for node_name, _ in palette])

        frame = Frame(tree, location=(x - 0.5, 0.5), label="ControlFrame",
                      node_height=GRID)
        frame.add(list(control.values()))
        return control

    # ------------------------------------------------------------------
    def _create_flight_path_frame(self, tree, control):
        """``Flight Path``: where each point of the molecule is, and when.

        The track comes in as a point cloud - one point per row of the csv,
        with the three columns as float attributes - which is turned back into
        a curve by placing each point at ``(X, Y, Z)`` and joining them in
        order.

        Point *i* of the molecule then sits at arc length
        ``HeadOffset - i * Spacing`` along that curve. ``Sample Curve`` reads a
        different length for every point, so one node serves the whole
        molecule, and lengths past either end of the track are clamped to its
        ends - which is why the track begins and ends off screen, where the
        pile-up cannot be seen.

        :return: the vector socket of the position each point should take.
        """
        csv = ImportCSV(tree, location=(2.5, 0.0), path=self.track_path,
                        name="TrackFile", label=DNA_TRACK_FILE,
                        node_height=GRID)

        columns = []
        for row, axis in enumerate("XYZ"):
            columns.append(NamedAttribute(tree, location=(2.5, -1.5 - 0.5 * row),
                                          data_type="FLOAT", name=axis,
                                          node_height=GRID))
        combine = CombineXYZ(tree, location=(4.0, -2.0), node_height=GRID,
                             x=columns[0].std_out, y=columns[1].std_out,
                             z=columns[2].std_out, name="TrackPoint")
        placed = SetPosition(tree, location=(5.5, 0.0),
                             geometry=csv.geometry_out,
                             position=combine.std_out, node_height=GRID,
                             name="PlaceTrack")
        track = PointsToCurve(tree, location=(7.0, 0.0), node_height=GRID,
                              name="Track")
        tree.links.new(placed.geometry_out, track.geometry_in)

        index = Index(tree, location=(4.0, -4.0), node_height=GRID,
                      name="FlightIndex")
        along = MathNode(tree, location=(5.5, -4.0), operation="MULTIPLY",
                         inputs0=index.std_out,
                         inputs1=control["Spacing"].std_out, node_height=GRID,
                         name="AlongMolecule")
        arc = MathNode(tree, location=(7.0, -4.0), operation="SUBTRACT",
                       inputs0=control["HeadOffset"].std_out,
                       inputs1=along.std_out, node_height=GRID,
                       name="ArcLength", label="ArcLength")

        sample = SampleCurve(tree, location=(8.5, -1.0), mode="LENGTH",
                             data_type="FLOAT", all_curves=True,
                             node_height=GRID, name="SampleTrack")
        tree.links.new(track.geometry_out, sample.geometry_in)
        tree.links.new(arc.std_out, sample.node.inputs["Length"])

        frame = Frame(tree, location=(2.0, 0.5), label="Flight Path",
                      node_height=GRID)
        frame.add([csv, combine, placed, track, index, along, arc,
                   sample] + columns)
        return sample.position_out

    # ------------------------------------------------------------------
    def _create_linear_structure_frame(self, tree, control, flight):
        """``LinearStructure``: the axis of the molecule, laid on the track.

        The line here exists only to carry ``pairs`` points in order - its own
        length and direction are thrown away by the Set Position, which moves
        every point onto the track instead of offsetting it.

        The normal is set explicitly to minimum twist. The default would flip
        where the track turns through the vertical, and the track does that
        twice on purpose.

        :return: the geometry socket of the molecule's axis.
        """
        line = CurveLine(tree, location=(12.0, 0.0), mode="DIRECTION",
                         direction=[1.0, 0.0, 0.0],
                         length=self.molecule_length, node_height=GRID,
                         name="Axis")
        resample = ResampleCurve(tree, location=(13.5, 0.0),
                                 curve=line.geometry_out, mode="Count",
                                 count=control["Pairs"].std_out,
                                 node_height=GRID, name="OnePointPerPair")
        placed = SetPosition(tree, location=(15.0, 0.0),
                             geometry=resample.geometry_out, position=flight,
                             node_height=GRID, name="OntoTrack")
        normal = SetCurveNormal(tree, location=(16.5, 0.0),
                                curve=placed.geometry_out,
                                mode="Minimum Twist", node_height=GRID,
                                name="StableNormal")

        frame = Frame(tree, location=(11.5, 0.5), label="LinearStructure",
                      node_height=GRID)
        frame.add([line, resample, placed, normal])
        return normal.geometry_out

    # ------------------------------------------------------------------
    def _create_unzipping_gate_frame(self, tree):
        """``Unzipping Gate``: how open each point of the molecule is.

        One number per point, ``0`` where the helix is to stay wound and ``1``
        where it is to be all the way open, read off the point's own ``y``:
        ``TRACK_Y_WOUND`` maps to 0, ``TRACK_Y_OPEN`` to 1, and the smoothstep
        in the middle is the fork. Everything the split does - the separation,
        the untwisting, the shrinking bases, the strand that peels away - is
        this one field crossfading between a wound branch and an open one, so
        they cannot come apart in different places.

        The gate is a *field* and deliberately not a selection. Every consumer
        blends with it rather than switching on it, which is the whole reason
        the transition can be smoothed at all - see ``Helix`` for the one place
        where blending the parameter instead of the position would have been a
        disaster.

        The track is what puts the gate where it is: ``y`` only ever climbs past
        ``TRACK_Y_WOUND`` on the last straight, so the fork stands at the exit
        of the turn and the molecule unzips itself by flying through it.

        :return: the float socket of the gate.
        """
        here = Position(tree, location=(17.0, -6.5), node_height=GRID,
                        name="GatePosition")
        depth = SeparateXYZ(tree, location=(18.5, -6.5),
                            vector=here.std_out, node_height=GRID,
                            name="Depth")
        gate = MapRange(tree, location=(20.0, -6.5), data_type="FLOAT",
                        interpolation_type="SMOOTHSTEP", value=depth.y,
                        from_min=TRACK_Y_WOUND, from_max=TRACK_Y_OPEN,
                        to_min=0.0, to_max=1.0, node_height=GRID,
                        name="Unzipped", label="Unzipped")

        frame = Frame(tree, location=(16.5, -6.0), label="Unzipping Gate",
                      node_height=GRID)
        frame.add([here, depth, gate])
        return gate.std_out

    # ------------------------------------------------------------------
    def _create_helix_frame(self, tree, control, axis, gate):
        """``Helix``: one line becomes two strands winding around each other.

        The two strands sit on opposite ends of a spoke that turns as it goes
        down the molecule. The spoke starts from a reference direction
        perpendicular to the track and is rotated about the tangent by
        ``index / TiltLength``; a spoke that turns steadily is what makes the
        pair wind. Raising ``TiltLength`` unwinds it, raising
        ``StrandSeparation`` pulls the two apart, and doing both at once is the
        molecule splitting.

        The reference direction is ``normalize(tangent x y)`` rather than the
        curve's own normal, and that choice is load-bearing. The curve normal
        is carried along the curve by minimum twist, so after two loops it
        points somewhere that depends on the whole history of the track - which
        would leave the strands separating in an arbitrary plane, quite
        possibly straight into the screen where the split would be invisible.
        The cross product has no history: wherever the molecule is running
        horizontally the spoke is vertical, so the split always opens upwards.
        It is well defined everywhere on this track because the track's tangent
        never comes near the y axis - it only ever leans a few degrees out of
        the x-z plane, to clear its own loops.

        Because the twist is driven by the *index* rather than by arc length,
        the helix is fixed in the molecule: it slides along the track without
        appearing to screw itself forward.

        All of that is built **twice**, once from the untouched
        ``Default*`` values and once from the dials the scene keyframes, and
        the gate mixes the two *offset vectors*. Mixing the vectors is not a
        detail; mixing the parameters instead - one Tilt Length that slides
        from 1.7 to 60 across the fork - is what the node editor invites, and
        it does not work. The angle is ``index / TiltLength``: at index 150
        that is 85 radians at the wound end of the fork and 2.5 at the open
        end, so a fork thirty pairs wide would have to spend thirteen full
        turns getting from one to the other. The strands would come out of it
        shredded. Two branches and a vector mix have no such term: each branch
        turns at its own gentle rate, and the crossfade between them decays the
        wound branch's radius to nothing while the open one grows, which is
        exactly what a strand coming off a helix looks like.

        The tangent and the two indices are captured because everything
        downstream needs them *after* the topology has changed: a field
        evaluated after ``Duplicate Elements`` no longer knows which strand it
        is on, and one evaluated after ``Curve to Mesh`` no longer knows which
        pair it came from.

        :param gate: the unzipping gate - 0 where the molecule is to stay
            wound, 1 where it is to be open.
        :return: dict with the geometry socket and the captured fields.
        """
        tangent = InputTangent(tree, location=(18.5, -2.5), node_height=GRID,
                               name="Tangent")
        unit_tangent = VectorMath(tree, location=(20.0, -2.5),
                                  operation="NORMALIZE",
                                  inputs0=tangent.std_out, node_height=GRID,
                                  name="UnitTangent")
        # The gate is captured here, on the axis, and every consumer reads the
        # captured copy. Read live instead, and each Set Position further down
        # would evaluate it on the points *the one before it had already
        # moved*: the second strand, pushed a separation off the axis, would
        # ask the gate about a y it does not have and get a different answer
        # from its own partner, which shows up as a pair that opens by
        # different amounts at its two ends. Freezing it is also the honest
        # statement of what the gate means - a base pair opens as a unit, at
        # the depth its axis is at, not at the depth either strand ends up.
        capture_tangent = CaptureAttribute(
            tree, location=(21.5, 0.0), domain="POINT", geometry=axis,
            items=[("Tangent", "FLOAT_VECTOR", unit_tangent.std_out),
                   ("Unzipped", "FLOAT", gate)],
            node_height=GRID, name="CaptureTangent")
        unzipped = capture_tangent["Unzipped"]

        pair_index = Index(tree, location=(21.5, -1.5), node_height=GRID,
                           name="PairIndexSource")
        capture_pair = CaptureAttribute(
            tree, location=(23.0, 0.0), domain="POINT",
            geometry=capture_tangent.geometry_out,
            items=[("PairIndex", "INT", pair_index.std_out)],
            node_height=GRID, name="CapturePairIndex")

        duplicate = DuplicateElements(tree, location=(24.5, 0.0),
                                      domain="SPLINE",
                                      geometry=capture_pair.geometry_out,
                                      amount=2, node_height=GRID,
                                      name="TwoStrands")
        capture_spline = CaptureAttribute(
            tree, location=(26.0, 0.0), domain="POINT",
            geometry=duplicate.geometry_out,
            items=[("SplineIndex", "INT", duplicate.duplicate_index)],
            node_height=GRID, name="CaptureSplineIndex")

        # The reference direction: perpendicular to the track and, wherever the
        # track runs level, vertical. ``y x tangent`` rather than the other way
        # round, and the order matters by the end of the shot. The curve runs
        # from the head of the molecule backwards, so on the last straight -
        # which is where the whole split happens - its tangent points left, and
        # this order is what puts the second strand *above* the first there.
        # The other order puts it below, and ``PeelHeight``, which lifts in
        # world z, would then drag it up through its own partner on the way
        # out of frame.
        reference = VectorMath(tree, location=(24.5, -2.5),
                               operation="CROSS_PRODUCT",
                               inputs0=[0.0, 1.0, 0.0],
                               inputs1=capture_tangent["Tangent"],
                               node_height=GRID, name="Reference")
        unit_reference = VectorMath(tree, location=(26.0, -2.5),
                                    operation="NORMALIZE",
                                    inputs0=reference.std_out,
                                    node_height=GRID, name="UnitReference")
        # the two branches. Each is "turn the reference spoke by index over a
        # tilt length, then push the second strand out along it by a
        # separation" - the wound one on the values the molecule was built
        # with, the open one on the values the scene dials.
        branches = []
        for row, (label, tilt, apart) in enumerate([
            ("Wound", "DefaultTiltLength", "DefaultStrandSeparation"),
            ("Open", "TiltLength", "StrandSeparation")]):
            twist = MathNode(tree, location=(24.5, -4.0 - 1.5 * row),
                             operation="DIVIDE",
                             inputs0=capture_pair["PairIndex"],
                             inputs1=control[tilt].std_out, node_height=GRID,
                             name=label + "Twist", label=label + "Twist")
            spoke = VectorRotate(tree, location=(26.0, -4.0 - 1.5 * row),
                                 rotation_type="AXIS_ANGLE",
                                 vector=unit_reference.std_out,
                                 axis=capture_tangent["Tangent"],
                                 angle=twist.std_out, node_height=GRID,
                                 name=label + "Spoke")
            offset = VectorMath(tree, location=(27.5, -4.0 - 1.5 * row),
                                operation="SCALE", inputs0=spoke.std_out,
                                node_height=GRID, name=label + "Separation")
            tree.links.new(control[apart].std_out,
                           offset.node.inputs["Scale"])
            branches += [twist, spoke, offset]

        separation = MixNode(tree, location=(29.0, -4.5), data_type="VECTOR",
                             factor=unzipped, input_a=branches[2].std_out,
                             input_b=branches[5].std_out, node_height=GRID,
                             name="Unzip", label="Unzip")

        # the second strand is pushed a whole separation along the spoke, then
        # both are pulled back half of it, so the axis of the molecule stays on
        # the track instead of drifting to one side of it
        push = SetPosition(tree, location=(30.5, 0.0),
                           geometry=capture_spline.geometry_out,
                           selection=duplicate.duplicate_index,
                           offset=separation.std_out, node_height=GRID,
                           name="PushSecondStrand")
        recenter_offset = VectorMath(tree, location=(30.5, -4.5),
                                     operation="SCALE",
                                     inputs0=separation.std_out,
                                     float_input=-0.5, node_height=GRID,
                                     name="HalfSeparation")
        recenter = SetPosition(tree, location=(32.0, 0.0),
                               geometry=push.geometry_out,
                               offset=recenter_offset.std_out,
                               node_height=GRID, name="RecenterAxis")

        frame = Frame(tree, location=(18.0, 0.5), label="Helix",
                      node_height=GRID)
        frame.add([tangent, unit_tangent, capture_tangent, pair_index,
                   capture_pair, duplicate, capture_spline, reference,
                   unit_reference, separation, push,
                   recenter_offset, recenter] + branches)
        return {
            "geometry": recenter.geometry_out,
            "Tangent": capture_tangent["Tangent"],
            "Unzipped": unzipped,
            "PairIndex": capture_pair["PairIndex"],
            "SplineIndex": capture_spline["SplineIndex"],
        }

    # ------------------------------------------------------------------
    def _create_strand_shift_frame(self, tree, control, helix):
        """``Strand Shift``: open the grooves, and later peel the two apart.

        Strand 0 slides one way along the tangent, strand 1 the other. Two
        strands exactly opposite each other would leave two identical grooves;
        sliding them apart makes one wide and one narrow, which is what the eye
        reads as DNA.

        ``PeelHeight`` then lifts strand 1 in world z. It is deliberately a
        world direction rather than the curve normal: which strand is the upper
        one changes from turn to turn while the helix is wound, so there is no
        "upper strand" to lift until the helix has unwound.

        The lift is gated, and has to be. It is 26 units, which is off the top
        of the frame; applied to the whole of strand 1 it would tear the part
        of the molecule that is still a double helix in half. Multiplied by the
        gate it lifts only what the fork has already let go of, so the second
        strand climbs out of the frame *from* the fork - which is the picture
        the shot is after in the first place.

        :return: the geometry socket of the finished pair of strands.
        """
        backward = MathNode(tree, location=(35.5, -3.0), operation="MULTIPLY",
                            inputs0=control["StrandShift"].std_out,
                            inputs1=-1.0, node_height=GRID,
                            name="BackwardShift")
        shift_first = VectorMath(tree, location=(37.0, -2.0),
                                 operation="SCALE", inputs0=helix["Tangent"],
                                 node_height=GRID, name="ShiftFirstStrand")
        tree.links.new(backward.std_out, shift_first.node.inputs["Scale"])
        first = BooleanMath(tree, location=(37.0, -3.5), operation="NOT",
                            inputs0=helix["SplineIndex"], node_height=GRID,
                            name="IsFirstStrand")
        move_first = SetPosition(tree, location=(38.5, 0.0),
                                 geometry=helix["geometry"],
                                 selection=first.std_out,
                                 offset=shift_first.std_out, node_height=GRID,
                                 name="MoveFirstStrand")

        shift_second = VectorMath(tree, location=(37.0, -1.0),
                                  operation="SCALE", inputs0=helix["Tangent"],
                                  node_height=GRID, name="ShiftSecondStrand")
        tree.links.new(control["StrandShift"].std_out,
                       shift_second.node.inputs["Scale"])
        move_second = SetPosition(tree, location=(40.0, 0.0),
                                  geometry=move_first.geometry_out,
                                  selection=helix["SplineIndex"],
                                  offset=shift_second.std_out,
                                  node_height=GRID, name="MoveSecondStrand")

        peel_where_open = MathNode(tree, location=(37.0, -4.5),
                                   operation="MULTIPLY",
                                   inputs0=control["PeelHeight"].std_out,
                                   inputs1=helix["Unzipped"], node_height=GRID,
                                   name="PeelWhereOpen")
        lift = CombineXYZ(tree, location=(38.5, -4.5), node_height=GRID,
                          z=peel_where_open.std_out, name="Lift")
        peel = SetPosition(tree, location=(41.5, 0.0),
                           geometry=move_second.geometry_out,
                           selection=helix["SplineIndex"],
                           offset=lift.std_out, node_height=GRID,
                           name="PeelSecondStrand")

        frame = Frame(tree, location=(35.0, 0.5), label="Strand Shift",
                      node_height=GRID)
        frame.add([backward, shift_first, first, move_first, shift_second,
                   move_second, peel_where_open, lift, peel])
        return peel.geometry_out

    # ------------------------------------------------------------------
    def _create_base_pair_coloring_frame(self, tree, control, helix, strands):
        """``Base Pair Coloring``: which of the four bases each point carries.

        ``BaseType = (PairIndex + SplineIndex) % 2 + 2 * random(PairIndex)``.

        The parity term is what makes a pair complementary: the two points of
        one pair differ in ``SplineIndex`` and therefore always land on
        different parities. The random bit is seeded by ``PairIndex`` alone, so
        both points of a pair draw the *same* bit and the pair is one of
        A-T / T-A / G-C / C-G rather than an arbitrary combination.

        The xml also built an RGBA colour here and stored it as ``C``. That is
        replaced by the material switches in the for-each zone - see the class
        docstring - so only the integer survives.

        :return: the geometry socket carrying the ``BaseType`` attribute.
        """
        purine = RandomValue(tree, location=(43.5, -2.0), data_type="BOOLEAN",
                             probability=0.5, seed=self.seed, node_height=GRID,
                             name="PurineOrPyrimidine")
        tree.links.new(helix["PairIndex"], purine.node.inputs["ID"])
        purine.node.inputs["Seed"].default_value = self.seed

        parity_sum = MathNode(tree, location=(43.5, -3.5), operation="ADD",
                              inputs0=helix["PairIndex"],
                              inputs1=helix["SplineIndex"], node_height=GRID,
                              name="PairPlusStrand")
        parity = IntegerMath(tree, location=(45.0, -3.5), operation="MODULO",
                             inputs0=parity_sum.std_out, inputs1=2,
                             node_height=GRID, name="Parity")
        upper = MathNode(tree, location=(45.0, -2.0), operation="MULTIPLY",
                         inputs0=purine.std_out, inputs1=2.0,
                         node_height=GRID, name="HighBit")
        base_type = MathNode(tree, location=(46.5, -3.0), operation="ADD",
                             inputs0=parity.std_out, inputs1=upper.std_out,
                             node_height=GRID, name="BaseType")
        store = StoredNamedAttribute(tree, location=(48.0, 0.0),
                                     data_type="INT", domain="POINT",
                                     name="BaseType", value=base_type.std_out,
                                     node_height=GRID)
        tree.links.new(strands, store.geometry_in)
        store.node.label = "BaseType"

        frame = Frame(tree, location=(43.0, 0.5), label="Base Pair Coloring",
                      node_height=GRID)
        frame.add([purine, parity_sum, parity, upper, base_type, store])
        return store.geometry_out

    # ------------------------------------------------------------------
    def _create_bases_and_pairing_frame(self, tree, control, helix, strands,
                                        colored):
        """``Bases And Pairing``: one base per point, aimed at its partner.

        A base is a short chain of 2..5 atoms - spheres on a mesh line, with
        the line itself swept to a tube for the bonds. How many atoms is what
        distinguishes the four ``BaseType`` values on screen, so a long base
        always faces a short one across the helix.

        The aim comes from sampling the position of the partner point.
        ``Duplicate Elements`` laid the second strand straight after the first,
        so the partner of point *i* is point *i + pairs* modulo *2 * pairs*,
        and the vector between them is what the base is rotated onto.

        The size of a base is gated along with everything else: bases that
        still reach across to a partner keep the size they were built with, and
        only the ones past the fork shrink to the stubs an unpaired strand is
        drawn with. It goes into the zone as an input item rather than being
        read inside it, because a field has to be evaluated where its geometry
        is - out here, once per element.

        :return: the geometry socket of all the bases.
        """
        pairs = Reroute(tree, location=(50.0, -6.0), node_height=GRID,
                        ins=control["Pairs"].std_out, name="PairsIn")

        index = Index(tree, location=(50.0, -4.5), node_height=GRID,
                      name="BaseIndex")
        stride = MathNode(tree, location=(51.0, -5.5), operation="ADD",
                          inputs0=pairs.std_out, inputs1=0.0,
                          node_height=GRID, name="Stride")
        partner_raw = MathNode(tree, location=(52.5, -4.5), operation="ADD",
                               inputs0=index.std_out, inputs1=stride.std_out,
                               node_height=GRID, name="PartnerRaw")
        total = MathNode(tree, location=(51.0, -6.5), operation="MULTIPLY",
                         inputs0=pairs.std_out, inputs1=2.0, node_height=GRID,
                         name="TotalPoints")
        partner = IntegerMath(tree, location=(54.0, -5.0), operation="MODULO",
                              inputs0=partner_raw.std_out,
                              inputs1=total.std_out, node_height=GRID,
                              name="PartnerIndex")

        here = Position(tree, location=(52.5, -3.0), node_height=GRID,
                        name="BasePosition")
        there = SampleIndex(tree, location=(55.5, -3.0),
                            data_type="FLOAT_VECTOR", domain="POINT",
                            geometry=strands, value=here.std_out,
                            index=partner.std_out, node_height=GRID,
                            name="PartnerPosition")
        across = VectorMath(tree, location=(57.0, -3.0), operation="SUBTRACT",
                            inputs0=there.std_out, inputs1=here.std_out,
                            node_height=GRID, name="AcrossTheHelix")
        aim = AlignRotationToVector(tree, location=(58.5, -3.0), axis="Z",
                                    pivot_axis="AUTO", vector=across.std_out,
                                    node_height=GRID, name="AimAtPartner")

        base_type = NamedAttribute(tree, location=(58.5, -1.0),
                                   data_type="INT", name="BaseType",
                                   node_height=GRID)
        base_type.node.label = "BaseType"

        size = MixNode(tree, location=(58.5, -6.5), data_type="FLOAT",
                       factor=helix["Unzipped"],
                       input_a=control["DefaultBaseSize"].std_out,
                       input_b=control["BaseSize"].std_out, node_height=GRID,
                       name="SizeHere", label="SizeHere")

        # one iteration per point of the two strands
        zone = ForEachZone(tree, location=(60.0, 0.0), domain="POINT",
                           node_width=12.5, geometry=colored,
                           node_height=GRID, name="ForEachBase")
        zone.add_socket("INT", "BaseType", value=base_type.std_out,
                        for_input=True)
        zone.add_socket("ROTATION", "Rotation", value=aim.std_out,
                        for_input=True)
        zone.add_socket("FLOAT", "BaseSize", value=size.std_out,
                        for_input=True)
        zone.foreach_output.location = (72.5 * GRID, 0)
        inside_type = zone.foreach_input.outputs["BaseType"]

        atoms = IndexSwitch(tree, location=(61.5, -2.0), data_type="INT",
                            index=inside_type, node_height=GRID,
                            name="AtomsPerBase")
        for _ in range(len(self.BASE_ATOMS) - 2):
            atoms.new_item()
        for slot, count in enumerate(self.BASE_ATOMS):
            atoms.node.inputs[slot + 1].default_value = count

        chain = MeshLine(tree, location=(63.0, -2.0), mode="END_POINTS",
                         count_mode="TOTAL", count=atoms.std_out,
                         start_location=[0.0, 0.0, 0.0], node_height=GRID,
                         name="BaseChain")
        chain.node.inputs["Offset"].default_value = [0.0, 0.0, 1.0]
        chain_out = Reroute(tree, location=(64.0, -2.0), node_height=GRID,
                            ins=chain.geometry_out, name="ChainOut")

        atom = IcoSphere(tree, location=(63.0, -0.5),
                         radius=self.BASE_ATOM_RADIUS, subdivisions=2,
                         node_height=GRID, name="Atom")
        atom_instances = InstanceOnPoints(tree, location=(65.0, -0.5),
                                          points=chain_out.geometry_out,
                                          instance=atom.geometry_out,
                                          node_height=GRID, name="Atoms")

        bonds = MeshToCurve(tree, location=(65.0, -2.0),
                            mesh=chain_out.geometry_out, node_height=GRID,
                            name="Bonds")
        bond_profile = CurveCircle(tree, location=(63.0, -3.5), mode="RADIUS",
                                   resolution=32,
                                   radius=self.BASE_BOND_RADIUS,
                                   node_height=GRID, name="BondProfile")
        bond_mesh = CurveToMesh(tree, location=(66.5, -2.0),
                                curve=bonds.geometry_out,
                                profile_curve=bond_profile.geometry_out,
                                fill_caps=False, node_height=GRID,
                                name="BondMesh")

        base = JoinGeometry(tree, location=(68.0, -1.0), node_height=GRID,
                            name="JoinBase")
        for piece in (atom_instances.geometry_out, bond_mesh.geometry_out):
            tree.links.new(piece, base.geometry_in)

        placed = InstanceOnPoints(tree, location=(69.5, 0.0),
                                  points=zone.element,
                                  instance=base.geometry_out,
                                  rotation=zone.foreach_input.outputs["Rotation"],
                                  node_height=GRID, name="PlaceBase")
        tree.links.new(zone.foreach_input.outputs["BaseSize"],
                       placed.node.inputs["Scale"])

        material = self._create_base_material_switches(tree, control,
                                                       inside_type)
        paint = SetMaterial(tree, location=(71.0, 0.0),
                            geometry=placed.geometry_out, material=material,
                            node_height=GRID, name="PaintBase")
        tree.links.new(paint.geometry_out,
                       zone.foreach_output.inputs["Geometry"])

        frame = Frame(tree, location=(49.5, 0.5), label="Bases And Pairing",
                      node_height=GRID)
        frame.add([pairs, index, stride, partner_raw, total, partner, here,
                   there, across, aim, base_type, size, zone, atoms, chain,
                   chain_out, atom, atom_instances, bonds, bond_profile,
                   bond_mesh, base, placed, paint] + self._switch_nodes)
        return zone.geometry_out

    # ------------------------------------------------------------------
    def _create_base_material_switches(self, tree, control, base_type):
        """The three switches that were RGBA in the xml, now materials.

        ``BaseType`` packs the two bits the xml switched on: its low bit is the
        pair parity, its high bit the purine/pyrimidine draw. Unpacking them
        here rather than carrying two more sockets into the zone keeps the zone
        interface to the two things it really needs.

        :return: the material socket of the base.
        """
        # only the four base colours are taken out of the palette here - a
        # Separate Bundle names what it wants, so the two backbone materials
        # travel past this frame untouched
        colors = SeparateBundle(
            tree, location=(59.5, -5.0), bundle=control["Palette"].std_out,
            items=[("Base%d" % i, "MATERIAL") for i in range(4)],
            node_height=GRID, name="BaseColors")

        parity = IntegerMath(tree, location=(61.5, -4.5), operation="MODULO",
                             inputs0=base_type, inputs1=2, node_height=GRID,
                             name="BaseParity")
        high = IntegerMath(tree, location=(61.5, -5.5), operation="DIVIDE",
                           inputs0=base_type, inputs1=2, node_height=GRID,
                           name="BaseHighBit")

        even = Switch(tree, location=(63.5, -4.5), input_type="MATERIAL",
                      switch=high.std_out, false=colors.out("Base0"),
                      true=colors.out("Base2"), node_height=GRID,
                      name="EvenPairMaterial")
        odd = Switch(tree, location=(63.5, -5.5), input_type="MATERIAL",
                     switch=high.std_out, false=colors.out("Base1"),
                     true=colors.out("Base3"), node_height=GRID,
                     name="OddPairMaterial")
        base = Switch(tree, location=(65.5, -5.0), input_type="MATERIAL",
                      switch=parity.std_out, false=even.std_out,
                      true=odd.std_out, node_height=GRID,
                      name="BaseMaterial")

        self._switch_nodes = [colors, parity, high, even, odd, base]
        return base.std_out

    # ------------------------------------------------------------------
    def _create_strand_geometry_frame(self, tree, control, strands):
        """``Strand Geometry``: the two backbones as solid tubes.

        The curve is swept to a tube for the sugar-phosphate backbone, and a
        sphere is dropped on every point so the backbone reads as a chain of
        atoms rather than a smooth pipe.

        :return: the geometry socket of both backbones.
        """
        curve = Reroute(tree, location=(74.0, -1.0), node_height=GRID,
                        ins=strands, name="StrandsIn")
        colors = SeparateBundle(
            tree, location=(75.5, -1.0), bundle=control["Palette"].std_out,
            items=[("Strand", "MATERIAL"), ("Molecule", "MATERIAL")],
            node_height=GRID, name="BackboneColors")

        profile = CurveCircle(tree, location=(74.0, -2.5), mode="RADIUS",
                              resolution=32, radius=self.BACKBONE_RADIUS,
                              node_height=GRID, name="BackboneProfile")
        tube = CurveToMesh(tree, location=(75.5, -2.5),
                           curve=curve.geometry_out,
                           profile_curve=profile.geometry_out,
                           fill_caps=False, node_height=GRID,
                           name="Backbone")
        tube_material = SetMaterial(tree, location=(77.0, -2.5),
                                    geometry=tube.geometry_out,
                                    material=colors.out("Strand"),
                                    node_height=GRID, name="PaintBackbone")

        atom = UVSphere(tree, location=(74.0, 0.5),
                        radius=self.BACKBONE_SPHERE_RADIUS, segments=32,
                        rings=16, node_height=GRID, name="BackboneAtom")
        atoms = InstanceOnPoints(tree, location=(75.5, 0.5),
                                 points=curve.geometry_out,
                                 instance=atom.geometry_out, node_height=GRID,
                                 name="BackboneAtoms")
        atom_material = SetMaterial(tree, location=(77.0, 0.5),
                                    geometry=atoms.geometry_out,
                                    material=colors.out("Molecule"),
                                    node_height=GRID, name="PaintAtoms")

        join = JoinGeometry(tree, location=(78.5, -1.0), node_height=GRID,
                            name="JoinBackbone")
        for piece in (tube_material.geometry_out, atom_material.geometry_out):
            tree.links.new(piece, join.geometry_in)
        smooth = SetShadeSmooth(tree, location=(80.0, -1.0),
                                geometry=join.geometry_out, node_height=GRID,
                                name="SmoothBackbone")

        frame = Frame(tree, location=(73.5, 0.5), label="Strand Geometry",
                      node_height=GRID)
        frame.add([curve, colors, profile, tube, tube_material, atom, atoms,
                   atom_material, join, smooth])
        return smooth.geometry_out


class RNAGridModifier(GeometryNodesModifier):
    """The 256 bytes as 256 little RNA strands, growing onto the screen column
    by column.

    A 16 x 16 grid of cells. Each cell holds one single strand of four bases -
    :class:`DNAModifier`'s molecule cut down to one backbone and four of its
    bases - and, right next to it, the number that strand spells out, written
    once in decimal and once in base 4.

    That is the whole point of the picture: a base is a digit. Four bases with
    four colours are a four-digit number in base 4, and four base-4 digits are
    exactly the 256 values of a byte, which is what a cell of a BFF tape holds.
    The grid is that dictionary, all 256 entries of it, in one shot.

    Three of the four colours are :class:`DNAModifier`'s, unchanged, so a
    strand here reads as a piece of the molecule the video opened with. The
    fourth is not: DNA's fourth base is thymine and RNA's is uracil, so the
    colour of DNA's fourth base is the one that is replaced. The atom counts
    :attr:`BASE_ATOMS` come over from :class:`DNAModifier` untouched - the two
    purines and the two pyrimidines are drawn with the same number of atoms
    here as there - while the radii are opened up (see the constants below): a
    cell of this grid is a diagram of a number, seen 256 at a time, not a
    molecule seen from a metre away.

    Digit order is the same in the strand and in the text: the top base of a
    strand is the most significant digit, the leftmost character of the base-4
    number, and the bottom base is the least significant. The four characters
    of the base-4 number are painted in their own base's colour, so that
    reading the strand and reading the number are visibly the same act.

    The layout is column-major - ``number = 16 * column + row`` - which makes
    each column one high nibble: a whole column shares the first *two* base-4
    digits, and only the bottom two bases change as the eye runs down it.

    The grid grows on rather than appearing: a column's cells scale up from
    nothing over ``growth_frames``, and each column starts ``speed_up`` times
    as long after its predecessor as the one before did. With the default
    ``speed_up`` below 1 the gaps shrink geometrically, so the first columns
    can be read one at a time and the last ones snap in almost together. Cells
    whose column has not started yet are not built at all (the for-each zone's
    selection), rather than built and scaled to zero.

    Everything is built in the x-z plane, facing a camera that looks along +y,
    like the rest of the flat scenes in this file.

    :param column_spacing: distance between the columns of the grid.
    :param row_spacing: distance between the rows.
    :param base_spacing: distance between two bases along one strand.
    :param helix_radius: how far the backbone is offset from the axis of the
        strand. This is what little is left of DNA's helix: the four backbone
        points sit on a circle of this radius, at ``base_twist`` to each other.
    :param base_twist: angle between one base and the next, in radians. Zero
        makes a flat comb, larger values wind the strand up.
    :param base_phase: the direction the middle of the fan of bases points, in
        radians, measured in the x-y plane. The default ``pi`` aims the bases
        left, away from the two numbers on the right of the cell.
    :param bond_length: distance between two atoms of one base.
    :param glyph_size: height of the characters of the two numbers.
    :param digit_spacing: distance between two digits of the base-4 number.
    :param text_x: x of the middle of both numbers, measured from the axis of
        the strand.
    :param line_gap: vertical distance between the decimal and the base-4 line.
    :param start_frame: frame at which the first column starts to grow.
    :param column_delay: frames between the first column and the second.
    :param speed_up: what that delay is multiplied by for every further column.
        Below 1 the grid gets faster as it fills, which is what is wanted; 1
        gives a constant rhythm.
    :param growth_frames: how long one column takes to grow to full size.
    :param base_colors: the four base materials, in digit order 0..3. The
        default is :attr:`BASE_COLORS`.
    :param strand_color: material of the swept backbone.
    :param molecule_color: material of the spheres sitting on the backbone.
    :param text_color: material of the decimal number.
    """

    COLUMNS = 16
    ROWS = 16
    #: bases on one strand, which is also the number of digits a byte has in
    #: base 4 - the two are the same thing here
    BASES = 4
    RADIX = 4
    #: position ``d`` of this string is the character of digit ``d``
    DIGITS = "0123"

    #: atom counts and base names in digit order: two purines, then the two
    #: pyrimidines, exactly as :class:`DNAModifier` draws them
    BASE_ATOMS = DNAModifier.BASE_ATOMS
    BASE_NAMES = ("A", "G", "C", "U")

    #: three of DNA's four base colours, and one that is not: DNA's fourth base
    #: is thymine, RNA's is uracil, so the fourth colour has to go
    BASE_COLORS = RNA_BASE_COLORS

    #: the backbone is DNA's, to the millimetre
    BACKBONE_RADIUS = DNAModifier.BACKBONE_RADIUS
    BACKBONE_SPHERE_RADIUS = DNAModifier.BACKBONE_SPHERE_RADIUS
    #: the bases are not: DNA draws them at ``base_size`` 0.4, which leaves an
    #: atom of radius 0.04 and a bond of 0.024. At the size a cell of a 16 x 16
    #: grid gets on screen that is a thread, and the colour it carries is what
    #: the whole picture is about, so a base here is drawn about twice as fat.
    BASE_ATOM_RADIUS = 0.075
    BASE_BOND_RADIUS = 0.045

    def __init__(self, column_spacing=3.1, row_spacing=1.75,
                 base_spacing=0.42, helix_radius=0.12, base_twist=0.45,
                 base_phase=pi, bond_length=1,
                 glyph_size=0.5, digit_spacing=0.26, text_x=0.85,
                 line_gap=0.62,
                 start_frame=1, column_delay=14.0, speed_up=0.88,
                 growth_frames=18.0,
                 base_colors=None, strand_color="gray_4",
                 molecule_color="gray_7", text_color="text",
                 name="RNAGrid", **kwargs):
        self.column_spacing = column_spacing
        self.row_spacing = row_spacing
        self.base_spacing = base_spacing
        self.helix_radius = helix_radius
        self.base_twist = base_twist
        self.base_phase = base_phase
        self.bond_length = bond_length
        self.glyph_size = glyph_size
        self.digit_spacing = digit_spacing
        self.text_x = text_x
        self.line_gap = line_gap
        self.start_frame = start_frame
        self.column_delay = column_delay
        self.speed_up = speed_up
        self.growth_frames = growth_frames
        self.base_colors = tuple(base_colors or self.BASE_COLORS)
        self.strand_color = strand_color
        self.molecule_color = molecule_color
        self.text_color = text_color
        self.kwargs = kwargs
        super().__init__(name=name, automatic_layout=False)

    # ------------------------------------------------------------------
    # what the scene needs to know about the timing, without taking the
    # formula in _create_cell_layout_frame apart
    # ------------------------------------------------------------------
    def column_start_frame(self, column):
        """The frame at which ``column`` starts to grow.

        ``start_frame + column_delay * (1 + q + ... + q^(column-1))`` with
        ``q = speed_up`` - the geometric sum the graph computes in one line of
        RPN, evaluated here in python so that a scene can hang a camera move or
        a caption off the same number.
        """
        q = self.speed_up
        if abs(q - 1.0) < 1e-9:
            gaps = column
        else:
            gaps = (1.0 - q ** column) / (1.0 - q)
        return self.start_frame + self.column_delay * gaps

    def reveal_end_frame(self):
        """The frame at which the last cell of the grid is at full size."""
        return self.column_start_frame(self.COLUMNS - 1) + self.growth_frames

    # ------------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        control = self._create_control_frame(tree)
        cells = self._create_cell_layout_frame(tree)
        grid = self._create_cell_frame(tree, control, cells)

        tree.links.new(grid, self.group_outputs.inputs["Geometry"])
        self.group_outputs.location = (13 * 200, 0)

    # ------------------------------------------------------------------
    def _create_control_frame(self, tree):
        """``Palette``: the seven materials the grid is drawn in.

        Unlike :class:`DNAModifier` there is no column of Input Value nodes
        beside them. Nothing here is a knob that survives to render time: the
        grid is 256 cells of four bases whatever happens, and every distance in
        it is folded into the RPN of a Function node at build time.

        :return: ``{name: node}``, keyed by the label the node carries in the
            editor.
        """
        x = -21.0
        palette = [("Base%d" % digit, color)
                   for digit, color in enumerate(self.base_colors)]
        palette += [("Strand", self.strand_color),
                    ("Molecule", self.molecule_color),
                    ("Text", self.text_color)]

        control = {}
        for row, (node_name, color) in enumerate(palette):
            # **self.kwargs carries things like `emission=0.6` through to every
            # material, as in the tape modifiers above - these scenes are lit
            # mostly by emission on a black background
            control[node_name] = InputMaterial(tree, location=(x, -0.4 * row),
                                               material=color, name=node_name,
                                               hide=True, **self.kwargs)
            self.materials.append(control[node_name].node.material)

        frame = Frame(tree, location=(x - 0.4, 0.6), label="Palette")
        frame.add(list(control.values()))
        return control

    # ------------------------------------------------------------------
    def _create_cell_layout_frame(self, tree):
        """``CellLayout``: where every cell is, how big it is, and whether it
        is there at all.

        One point per cell, and one Function node that turns the index of that
        point into the three things the for-each zone downstream needs:

        ``Location``
            ``column = index // ROWS`` across, ``row = index % ROWS`` down.
            Column-major, so that a column of the grid is 16 consecutive
            numbers and the whole column shares its top two base-4 digits.
        ``Scale``
            zero until the cell's column starts, then smoothstep to one over
            ``growth_frames``. This is what "grows onto the screen" means: the
            cell is built at full size and scaled about its own origin.
        ``Visible``
            true once the column has started. It becomes the selection of the
            zone, so a cell that is not on screen yet is not built - 256 cells
            of four bases and six little glyphs is enough geometry that it is
            worth not building the ones nobody can see.

        The reveal times are a geometric series: column *c* starts at
        ``start_frame + column_delay * (1 - q^c)/(1 - q)``, ``q = speed_up``.
        Successive gaps are ``column_delay * q^c``, so with ``q < 1`` the grid
        accelerates as it fills. :meth:`column_start_frame` is the same formula
        in python.

        :return: ``{name: socket}`` - the point cloud, the index field, and the
            three outputs.
        """
        x = -19.0
        count = self.COLUMNS * self.ROWS

        points = Points(tree, location=(x, 0.0), count=count, name="Cells")
        index = Index(tree, location=(x, -1.2), name="CellIndex", hide=True)
        # no simulation zone anywhere here: what a cell looks like depends only
        # on the current frame, never on the previous one, so the frame number
        # is read straight off the scene - the same choice SoupWatcherModifier
        # makes above
        now = SceneTime(tree, location=(x, -1.8), std_out="Frame", name="Now",
                        hide=True)

        q = float(self.speed_up)
        delay = repr(float(self.column_delay))
        first = repr(float(self.start_frame))
        if abs(q - 1.0) < 1e-9:
            # a constant rhythm: c gaps before column c
            gaps = "col"
        elif q < 1.0:
            # (1 - q^c)/(1 - q), written so that both literals stay positive
            gaps = "1,%s,col,**,-,%s,/" % (repr(q), repr(1.0 - q))
        else:
            # the same sum for a grid that slows down instead
            gaps = "%s,col,**,1,-,%s,/" % (repr(q), repr(q - 1.0))
        start = "%s,%s,*,%s,+" % (gaps, delay, first)

        layout = make_function(
            tree, name="CellLayout",
            aux_functions={
                "col": "index,%d,/,floor" % self.ROWS,
                "row": "index,%d,%%" % self.ROWS,
                "start": start,
                "grow": "frame,start,-,%s,/,0,max,1,min"
                        % repr(float(self.growth_frames)),
                # 3g^2 - 2g^3, so a cell arrives and settles instead of
                # stopping dead at full size
                "smooth": "grow,grow,*,3,2,grow,*,-,*",
            },
            functions={
                "Location": [
                    "col,%s,-,%s,*" % (repr((self.COLUMNS - 1) / 2),
                                       repr(self.column_spacing)),
                    "0",
                    "%s,row,-,%s,*" % (repr((self.ROWS - 1) / 2),
                                       repr(self.row_spacing)),
                ],
                "Scale": ["smooth", "smooth", "smooth"],
                "Visible": "frame,start,>",
            },
            inputs=["index", "frame"],
            outputs=["Location", "Scale", "Visible"],
            vectors=["Location", "Scale"], booleans=["Visible"],
            scalars=["index", "frame", "col", "row", "start", "grow", "smooth"],
            hide=True, location=(x + 1.6, -1.2))
        tree.links.new(index.std_out, layout.inputs["index"])
        tree.links.new(now.std_out, layout.inputs["frame"])

        frame = Frame(tree, location=(x - 0.4, 0.6), label="CellLayout")
        frame.add([points, index, now, layout])
        return {
            "geometry": points.geometry_out,
            "index": index.std_out,
            "Location": layout.outputs["Location"],
            "Scale": layout.outputs["Scale"],
            "Visible": layout.outputs["Visible"],
        }

    # ------------------------------------------------------------------
    def _create_cell_frame(self, tree, control, cells):
        """``Cell``: one iteration of the for-each zone is one cell.

        Everything inside is built in the cell's own coordinates, with the axis
        of the strand at the origin, and the last two nodes of the zone scale
        it by ``Scale`` and move it to ``Location``. Transform Geometry scales
        about the origin before it translates, so a cell grows out of the point
        it will end up on rather than out of the middle of the grid.

        The zone carries the cell's number in as ``Number``. Inside, that is a
        single value rather than a field, which is what makes the numbers
        possible at all: ``Value to String`` and ``String to Curves`` have no
        field inputs, so a number can only be written by a graph that handles
        one cell at a time.

        :return: the geometry socket of the whole grid.
        """
        zone = ForEachZone(tree, location=(-16, 0), domain="POINT",
                           node_width=25, geometry=cells["geometry"],
                           selection=cells["Visible"], name="Cell")
        zone.add_socket("INT", "Number", value=cells["index"], for_input=True)
        zone.add_socket("VECTOR", "Location", value=cells["Location"],
                        for_input=True)
        zone.add_socket("VECTOR", "Scale", value=cells["Scale"],
                        for_input=True)
        number = zone.foreach_input.outputs["Number"]

        aim, aim_nodes = self._create_base_directions(tree)
        strand, backbone, strand_nodes = self._create_strand(tree, control, aim)
        bases, base_nodes = self._create_bases(tree, control, backbone, aim,
                                               number)
        labels, label_nodes = self._create_labels(tree, control, number)

        join = JoinGeometry(tree, location=(6.5, 0.0), name="JoinCell")
        for piece in [strand] + bases + labels:
            tree.links.new(piece, join.geometry_in)
        place = TransformGeometry(tree, location=(8.0, 0.0),
                                  geometry=join.geometry_out,
                                  translation=zone.foreach_input.outputs["Location"],
                                  scale=zone.foreach_input.outputs["Scale"],
                                  name="PlaceCell")
        tree.links.new(place.geometry_out, zone.foreach_output.inputs["Geometry"])

        frame = Frame(tree, location=(-16.4, 0.6), label="Cell")
        frame.add([zone, join, place] + aim_nodes + strand_nodes + base_nodes
                  + label_nodes)
        return zone.geometry_out

    # ------------------------------------------------------------------
    def _create_base_directions(self, tree):
        """Which way base ``slot`` points, and where its backbone atom sits.

        ``theta = base_phase + (slot - (BASES-1)/2) * base_twist`` - the four
        bases fanned symmetrically about ``base_phase`` - and from it a unit
        vector in the x-y plane for the base to be aimed along, and the same
        vector at ``helix_radius`` for the backbone point to be pushed to.

        This is where :class:`DNAModifier`'s helix ends up. There it is built
        the honest way, by tilting the curve so that its own normal turns; a
        strand of four points does not need a curve normal to be persuaded to
        wind, and computing the direction outright means the bases can be aimed
        at it without a Capture Attribute to carry it across the topology
        change.

        :return: ``(node, nodes)`` - the Function node, with outputs ``offset``
            and ``direction``, and the list to frame.
        """
        half = repr((self.BASES - 1) / 2)
        radius = repr(self.helix_radius)
        aim = make_function(
            tree, name="BaseDirection",
            aux_functions={
                "theta": "slot,%s,-,%s,*,%s,+" % (half, repr(self.base_twist),
                                                  repr(self.base_phase)),
            },
            functions={
                "direction": ["theta,cos", "theta,sin", "0"],
                "offset": ["theta,cos,%s,*" % radius,
                           "theta,sin,%s,*" % radius, "0"],
            },
            inputs=["slot"], outputs=["direction", "offset"],
            vectors=["direction", "offset"], scalars=["slot", "theta"],
            hide=True, location=(-13.5, 3.4))
        slot = Index(tree, location=(-14.6, 3.4), name="BaseSlot", hide=True)
        tree.links.new(slot.std_out, aim.inputs["slot"])
        return aim, [slot, aim]

    # ------------------------------------------------------------------
    def _create_strand(self, tree, control, aim):
        """``Strand``: the backbone of one cell, swept to a tube.

        A line of ``BASES`` points along +z, each pushed out to
        ``helix_radius`` in the direction its base points, so the backbone
        zigzags around the axis the way a real one winds. The line is a curve
        all the way through - Set Position keeps whatever it is given - so it
        can be swept with Curve to Mesh, and its points are also what the bases
        and the backbone spheres are instanced on.

        :return: ``(geometry, backbone, nodes)`` - the painted strand, the
            curve whose points carry the four bases, and the list to frame.
        """
        span = (self.BASES - 1) * self.base_spacing
        axis = CurveLine(tree, location=(-13.5, 2.2), mode="POINTS",
                         start=Vector([0.0, 0.0, -0.5 * span]),
                         end=Vector([0.0, 0.0, 0.5 * span]), name="StrandAxis")
        points = ResampleCurve(tree, location=(-12.4, 2.2), mode="Count",
                               curve=axis.geometry_out, count=self.BASES,
                               name="OnePointPerBase")
        backbone = SetPosition(tree, location=(-11.3, 2.2),
                               geometry=points.geometry_out,
                               offset=aim.outputs["offset"], name="Wind")

        profile = CurveCircle(tree, location=(-11.3, 1.4), mode="RADIUS",
                              resolution=12, radius=self.BACKBONE_RADIUS,
                              name="BackboneProfile", hide=True)
        tube = CurveToMesh(tree, location=(-10.2, 2.0),
                           curve=backbone.geometry_out,
                           profile_curve=profile.geometry_out, fill_caps=False,
                           name="Backbone")
        tube_material = SetMaterial(tree, location=(-9.1, 2.0),
                                    geometry=tube.geometry_out,
                                    material=control["Strand"].std_out,
                                    name="PaintBackbone")

        atom = UVSphere(tree, location=(-11.3, 3.0),
                        radius=self.BACKBONE_SPHERE_RADIUS, segments=16,
                        rings=8, name="BackboneAtom", hide=True)
        atoms = InstanceOnPoints(tree, location=(-10.2, 3.0),
                                 points=backbone.geometry_out,
                                 instance=atom.geometry_out,
                                 name="BackboneAtoms")
        atom_material = SetMaterial(tree, location=(-9.1, 3.0),
                                    geometry=atoms.geometry_out,
                                    material=control["Molecule"].std_out,
                                    name="PaintAtoms")

        join = JoinGeometry(tree, location=(-8.0, 2.5), name="JoinStrand")
        for piece in (tube_material.geometry_out, atom_material.geometry_out):
            tree.links.new(piece, join.geometry_in)
        smooth = SetShadeSmooth(tree, location=(-6.9, 2.5),
                                geometry=join.geometry_out,
                                name="SmoothStrand")

        nodes = [axis, points, backbone, profile, tube, tube_material, atom,
                 atoms, atom_material, join, smooth]
        return smooth.geometry_out, backbone.geometry_out, nodes

    # ------------------------------------------------------------------
    def _create_bases(self, tree, control, backbone, aim, number):
        """``Bases``: the four base molecules, one per backbone point.

        The base of slot *j* is digit *j* of the number in base 4, counting
        from the bottom of the strand: ``digit = (number // 4^j) % 4``. Four
        of them, bottom to top, are the number's base-4 digits least
        significant first, so the strand read from the top spells the number
        the way it is written.

        Which base a point gets cannot be an Index Switch, because a base is
        geometry rather than a value: all four are built and each is instanced
        with a *selection*, ``digit == d``. The four selections are disjoint by
        construction, so every point gets exactly one base.

        A base itself is :class:`DNAModifier`'s: ``BASE_ATOMS[d]`` spheres on a
        chain, with the chain swept to a thin tube for the bonds, aimed away
        from the backbone by aligning its +z with the direction the slot
        points.

        :return: ``(geometries, nodes)`` - one geometry socket per base type
            and the list to frame.
        """
        selector = make_function(
            tree, name="BaseSelector",
            aux_functions={
                "digit": "number,%d,slot,**,/,floor,%d,%%"
                         % (self.RADIX, self.RADIX),
            },
            functions={"IsBase%d" % d: "digit,%d,=" % d
                       for d in range(self.RADIX)},
            inputs=["number", "slot"],
            outputs=["IsBase%d" % d for d in range(self.RADIX)],
            booleans=["IsBase%d" % d for d in range(self.RADIX)],
            scalars=["number", "slot", "digit"],
            hide=True, location=(-13.5, 0.8))
        slot = Index(tree, location=(-14.6, 0.8), name="DigitSlot", hide=True)
        tree.links.new(slot.std_out, selector.inputs["slot"])
        tree.links.new(number, selector.inputs["number"])

        # the base points away from the backbone: its chain is built along +z
        # and Align Rotation to Vector turns that onto the slot's direction
        rotation = AlignRotationToVector(tree, location=(-13.5, 0.0), axis="Z",
                                         pivot_axis="AUTO",
                                         vector=aim.outputs["direction"],
                                         name="AimBase")

        # one sphere and one bond profile for all four base types
        atom = IcoSphere(tree, location=(-12.4, 0.0),
                         radius=self.BASE_ATOM_RADIUS, subdivisions=2,
                         name="BaseAtom", hide=True)
        profile = CurveCircle(tree, location=(-12.4, -0.4), mode="RADIUS",
                              resolution=8, radius=self.BASE_BOND_RADIUS,
                              name="BondProfile", hide=True)

        nodes = [slot, selector, rotation, atom, profile]
        geometries = []
        for d, atoms_per_base in enumerate(self.BASE_ATOMS):
            y = -1.2 - 1.4 * d
            name = self.BASE_NAMES[d]
            chain = MeshLine(tree, location=(-12.4, y), mode="END_POINTS",
                             count_mode="TOTAL", count=atoms_per_base,
                             start_location=Vector([0.0, 0.0, 0.0]),
                             end_location=Vector([0.0, 0.0, self.bond_length]),
                             name="Chain" + name)
            spheres = InstanceOnPoints(tree, location=(-11.3, y),
                                       points=chain.geometry_out,
                                       instance=atom.geometry_out,
                                       name="Atoms" + name)
            bonds = MeshToCurve(tree, location=(-11.3, y - 0.5),
                                mesh=chain.geometry_out, name="Bonds" + name,
                                hide=True)
            bond_mesh = CurveToMesh(tree, location=(-10.2, y - 0.5),
                                    curve=bonds.geometry_out,
                                    profile_curve=profile.geometry_out,
                                    fill_caps=False, name="BondMesh" + name)
            base = JoinGeometry(tree, location=(-9.1, y - 0.2),
                                name="JoinBase" + name)
            for piece in (spheres.geometry_out, bond_mesh.geometry_out):
                tree.links.new(piece, base.geometry_in)
            painted = SetMaterial(tree, location=(-8.0, y - 0.2),
                                  geometry=base.geometry_out,
                                  material=control["Base%d" % d].std_out,
                                  name="Paint" + name)
            placed = InstanceOnPoints(tree, location=(-6.9, y - 0.2),
                                      points=backbone,
                                      selection=selector.outputs["IsBase%d" % d],
                                      instance=painted.geometry_out,
                                      rotation=rotation.std_out,
                                      name="Place" + name)
            nodes += [chain, spheres, bonds, bond_mesh, base, painted, placed]
            geometries.append(placed.geometry_out)

        return geometries, nodes

    # ------------------------------------------------------------------
    def _create_labels(self, tree, control, number):
        """``Numbers``: the same byte written twice, next to its strand.

        The decimal on the upper line, the four base-4 digits on the lower one,
        and every one of those four characters painted in the colour of the
        base that stands for it - so the row of four colours down the strand
        and the row of four characters across the text are the same row, said
        twice.

        The digits are single values, not a field: inside the for-each zone the
        cell's number is one number, so ``digit_k = (number // 4^(3-k)) % 4``
        is too, and it can drive a Slice String into ``DIGITS`` and an Index
        Switch over the four materials. Both would be impossible one level up,
        where the number is a field over 256 cells.

        The text comes out of String to Curves lying in the x-y plane; the
        quarter turn about x stands it up in the x-z plane the rest of the cell
        is built in.

        :return: ``(geometries, nodes)`` - one geometry socket per line of
            text and the list to frame.
        """
        digits = make_function(
            tree, name="Base4Digits",
            functions={"Digit%d" % k: "number,%d,/,floor,%d,%%"
                                      % (self.RADIX ** (self.BASES - 1 - k),
                                         self.RADIX)
                       for k in range(self.BASES)},
            inputs=["number"],
            outputs=["Digit%d" % k for k in range(self.BASES)],
            integers=["Digit%d" % k for k in range(self.BASES)],
            scalars=["number"],
            hide=True, location=(-13.5, -7.0))
        tree.links.new(number, digits.inputs["number"])

        nodes = [digits]
        geometries = []

        # the decimal, centred on the upper line
        decimal = ValueToString(tree, location=(-12.4, -6.2), data_type="INT",
                                value=number, name="Decimal", hide=True)
        curves = StringToCurves(tree, location=(-11.3, -6.2),
                                string=decimal.std_out, size=2 * self.glyph_size,
                                align_x="CENTER", align_y="MIDDLE",
                                name="DecimalCurves")
        realize = RealizeInstances(tree, location=(-10.2, -6.2))
        fill = FillCurve(tree, location=(-9.1, -6.2), mode="N-gons")
        painted = SetMaterial(tree, location=(-8.0, -6.2),
                              material=control["Text"].std_out,
                              name="PaintDecimal")
        placed = TransformGeometry(
            tree, location=(-6.9, -6.2),
            translation=Vector([self.text_x, 0.0, 0.5 * self.line_gap]),
            rotation=[pi / 2, 0.0, 0.0], name="PlaceDecimal")
        create_geometry_line(tree, [realize, fill, painted, placed],
                             ins=curves.geometry_out)
        nodes += [decimal, curves, realize, fill, painted, placed]
        geometries.append(placed.geometry_out)

        # and the four base-4 digits below it, most significant first
        left = -0.5 * (self.BASES - 1) * self.digit_spacing
        for k in range(self.BASES):
            y = -7.6 - 1.0 * k
            digit = digits.outputs["Digit%d" % k]
            character = SliceString(tree, location=(-12.4, y),
                                    string=self.DIGITS, position=digit,
                                    length=1, name="Digit%d" % k, hide=True)
            # a material cannot be picked by a selection the way a base is -
            # Set Material takes one - so here the choice really is a switch
            color = IndexSwitch(tree, location=(-12.4, y - 0.4),
                                data_type="MATERIAL", index=digit,
                                name="Digit%dColor" % k, hide=True)
            for d in range(self.RADIX):
                color.add_item(socket=control["Base%d" % d].std_out)

            curves = StringToCurves(tree, location=(-11.3, y),
                                    string=character.std_out,
                                    size=self.glyph_size, align_x="CENTER",
                                    align_y="MIDDLE",
                                    name="Digit%dCurves" % k)
            realize = RealizeInstances(tree, location=(-10.2, y))
            fill = FillCurve(tree, location=(-9.1, y), mode="N-gons")
            painted = SetMaterial(tree, location=(-8.0, y),
                                  material=color.std_out,
                                  name="PaintDigit%d" % k)
            placed = TransformGeometry(
                tree, location=(-6.9, y),
                translation=Vector([self.text_x + left + k * self.digit_spacing,
                                    0.0, -0.5 * self.line_gap]),
                rotation=[pi / 2, 0.0, 0.0], name="PlaceDigit%d" % k)
            create_geometry_line(tree, [realize, fill, painted, placed],
                                 ins=curves.geometry_out)
            nodes += [character, color, curves, realize, fill, painted, placed]
            geometries.append(placed.geometry_out)

        return geometries, nodes


#: how many circles the logo's chain is made of - the ``N`` the graph builds
#: the outline from, and the only thing that decides what shape it is
LOGO_CIRCLES = 4
#: how many points the graph draws the outline with. A geometry budget, baked
#: into the node tree rather than a dial: it is the polyline the strand is
#: sampled off, so it only has to be much finer than the base spacing, and each
#: of the ``2 * N`` half circles gets the same share of it whatever its size.
LOGO_SAMPLES = 2000

#: how the complex plane is laid into blender's axes. Per plane: which axis
#: takes the real part, which takes the imaginary part, which is left over, and
#: the unit normal of the plane. ``"xy"`` is
#: :func:`utils.utils.z2vec`'s default and what the rest of the Apollonian
#: video draws in; ``"xz"`` is ``z2vec(..., z_dir=True)``, the plane the flat
#: scenes of the BFF video use.
LOGO_PLANES = {
    "xy": ((0, 1, 2), (0.0, 0.0, 1.0)),
    "xz": ((0, 2, 1), (0.0, -1.0, 0.0)),
}


def logo_outline(n=LOGO_CIRCLES, samples=LOGO_SAMPLES):
    """The logo's outline, as a closed smooth curve.

    **Nothing in the modifier calls this.**
    :meth:`RNALogoModifier._create_logo_path_frame` builds the same curve out
    of ``Math`` nodes, so that ``N`` is a socket rather than a decision taken
    at build time. This is the same formula in python: the reference the graph
    is checked against, and the readable statement of what it draws.

    :func:`objects.logo.logo_curve` is the logo drawn in one stroke: a chain of
    tangent circles - the big one in the middle and smaller ones going down
    each side - as the image under ``z -> 1/conj(z)`` of a row of half circles.
    The parameter runs over ``[-pi, pi]`` and the curve closes there *exactly*
    (``logo_curve(-pi, n) == logo_curve(pi, n)`` to the last bit) for whole
    ``n``, which is what the track downstream needs.
    ``video_bff/scene_bff.py``'s ``branding`` draws the same thing with
    :class:`~objects.curve.Curve`, over a domain that runs a tenth past ``pi``;
    that overshoot is harmless for a curve that is only looked at and is left
    out here, since it would leave the loop open.

    ``n`` is how much logo there is. Every circle is tangent to the next and
    each is smaller than the last, so raising it adds detail at the bottom of
    the picture and nowhere else: at 4 the chain ends in a circle a sixth of
    the height, at 20 in one a hundredth. :func:`logo_radii` has them all,
    exactly.

    For a strand riding on the outline, 4 is about as far as it goes, and the
    limit is the molecule rather than the curve. A base is at least
    ``spacing`` from the next, so on a circle small enough the bases are wider
    apart than the circle is round and what renders is a tangle rather than a
    strand. At 4 every circle including the last is clean; by 8 the smallest
    two have gone; by 20 the bottom of the logo is a knot. The outline itself
    is fine at any ``n`` - it is only what is riding on it that runs out of
    room.

    The result is halved, which puts it in ``x`` in ``[-0.5, 0.5]`` and ``y``
    in ``[0, 1]`` - the box the Apollonian limit set used to arrive in, so that
    ``scale`` still means the diameter of the whole logo and a camera framed
    for one is framed for the other.

    :param n: circles per side of the chain.
    :param samples: how many points the outline is drawn with. The last repeats
        the first, exactly, so the polyline is closed.
    :return: the outline as an array of complex numbers.
    """
    return 0.5 * logo_curve(np.linspace(-np.pi, np.pi, samples + 1), n)


def logo_radii(n=LOGO_CIRCLES):
    """The exact radius of every circle the outline is made of.

    The outline is not merely smooth, it is *circles*, and that is worth having
    in closed form rather than measuring. Before the inversion, ``logo_curve``
    is a row of ``n`` half circles all of radius ``1/4``, the ``k``-th centred
    at ``c_k = (n/4 - k/2) + 3i/4``; the parameter runs down the row and back
    up it, so each circle is traversed once, in two halves. Inversion in the
    unit circle takes a circle to a circle, of radius
    ``r / | |c|^2 - r^2 |`` - so every arc of the logo has a radius that can be
    written down, and there is nothing to fit.

    That is what replaced the least-squares circle fit this file used to run
    over a window of the source points. The fit was there because the source
    was a fractal and had no curvature to read off; on the outline it was
    answering a question that has an exact answer, and answering it slightly
    wrong - it reported 3.27 for the big circle where the truth is 3.

    Halved along with the outline itself, so these are radii of the curve
    :func:`logo_outline` hands back, before ``scale``.

    :param n: how many circles the chain has.
    :return: the ``n`` radii, in the order ``k = 0 .. n-1`` - smallest first,
        up to the big one in the middle of the chain and back down.
    """
    k = np.arange(n)
    centre = 0.25 * n - 0.5 * k
    return 0.125 / np.abs(centre ** 2 + 0.5625 - 0.0625)


def logo_track_length(n=LOGO_CIRCLES, scale=1.0):
    """How far it is once round the logo.

    Every circle is gone round exactly once, so the arc length is the sum of
    their circumferences and :func:`logo_radii` is all it takes. The graph gets
    the same number from a ``Curve Length`` node on the polyline it draws,
    which agrees with this to the fifth decimal at the default ``samples``;
    this one exists so that a scene can know how far to ramp ``HeadOffset``
    without evaluating the modifier.

    :param n: how many circles the chain has.
    :param scale: the diameter the logo is drawn at.
    :return: the arc length of one lap.
    """
    return float(2.0 * np.pi * scale * logo_radii(n).sum())


def logo_track_bases(n=LOGO_CIRCLES, bases_per_circle=17):
    """How many bases one lap of the logo holds.

    Every arc of the chain is a whole circle and each gets
    ``bases_per_circle`` of them whatever its size, so the count is a
    multiplication - which is the point of stepping by the circumference
    rather than by a fixed distance.

    :param n: how many circles the chain has.
    :param bases_per_circle: bases on each of them.
    :return: bases in a lap.
    """
    return int(n * bases_per_circle)


def _in_frame(origin):
    """Where to put a node that is going to end up inside a frame.

    Locations in this file are *absolute*: :meth:`Frame.add` parents a node
    after it has been placed, and blender rewrites ``location`` to be relative
    while leaving the node where it was on screen. So the contents of a frame
    have to be written down in absolute coordinates, and what the editor - and
    an exported xml - shows for a framed node is the difference between the
    two. This turns the second back into the first, so that the numbers in the
    code are the numbers in the xml.

    :param origin: the frame's own location.
    :return: a function of a node's location within that frame.
    """
    return lambda x, y: (origin[0] + x, origin[1] + y)


class RNALogoModifier(GeometryNodesModifier):
    """A single strand of RNA that draws the logo.

    :class:`DNAModifier` flies a double helix along a track read from a csv;
    this grows one strand along a track the graph *draws*, out of two integers.
    Nothing is read from disk and nothing is measured: the logo is a chain of
    circles, so where a base goes, how sharply the curve turns under it and
    which way it should point all have closed forms, and ``LogoCurve`` hands
    back all three.

    The molecule is built in seven frames:

    ``ControlFrame``
        Five numbers and the palette. ``Progress`` is the whole of the
        choreography - it is both how far the head has got and how many bases
        are behind it, so one ramp draws the logo.
    ``CreateLogo Intrinsic Resolution``
        ``PointCount`` points with :func:`objects.logo.logo_curve` evaluated
        over them as a field. This is the polyline everything downstream reads;
        its resolution is "intrinsic" in the sense that it is the same all the
        way round, in the *parameter*, which is not the same as being even along
        the curve - that is what the next two frames are for.
    ``DistanceToNextPoint``
        How far it is from each point of the outline to the next.
    ``PointSelection``
        The step a base is entitled to here - one ``BasesPerCircle``-th of the
        way round the circle it is standing on - and the running total of how
        many bases the outline has been worth so far. Laying that total out
        along the x axis gives the ``Ruler``, a curve whose *arc length is the
        base number*.
    ``GrowTheCurve``
        Base *i* sits at station ``Progress - i``, wrapped into one lap, and
        one Sample Curve on the ruler turns that back into a position, a radius
        and a normal.
    ``Strand``
        ``Progress`` points, moved onto the logo.
    ``Bases`` and ``Strand Geometry``
        A base per point, aimed down the normal, and the backbone swept to a
        tube - all three scaled by the radius, so the molecule is the same
        shape on the big circles and the small ones.

    **Why the sampling is not evenly spaced.** A base every fixed distance puts
    twenty of them on a circle only once its radius passes about a unit, and
    the logo's circles run from a radius of 3 down to a hundredth of that - so
    the small ones would get a handful of bases, then three, then one. Stepping
    by ``2 pi r / BasesPerCircle`` instead gives every circle the same number
    however small it is. The cost is that where a base sits is then a running
    total, which no field can work out; ``PointSelection`` pays it with an
    Accumulate Field, once, over the outline.

    :param n: circles per side of the chain - how much logo there is.
    :param scale: the diameter the logo is drawn at.
    :param progress: how far round the head has got, in bases, and therefore
        how many bases there are. Animate this and the logo draws itself.
    :param point_count: points the outline is drawn with. It only has to be
        much finer than the bases; each of the ``2 n`` half circles gets the
        same share of it whatever its size.
    :param bases_per_circle: bases on every circle of the chain, whatever its
        radius.
    :param axis_length: length of the line the strand's points are resampled
        from. Thrown away by the Set Position that moves them onto the logo.
    :param backbone_scale: backbone bead size per unit of radius.
    :param tube_fraction: the swept tube's scale as a fraction of the beads' -
        and, since the bases take the same number, their size too.
    :param base_colors: the four base materials, in ``BaseType`` order.
    :param strand_color: material of the swept backbone.
    :param molecule_color: material of the spheres sitting on the backbone.
    :param seed: which strand of RNA this is. Changing it deals the bases again.
    """

    #: the two purines and the two pyrimidines, drawn with as many atoms as
    #: :class:`DNAModifier` draws them with
    BASE_ATOMS = DNAModifier.BASE_ATOMS
    BASE_NAMES = ("A", "G", "C", "U")
    #: three of DNA's four base colours and one that is not - DNA's fourth base
    #: is thymine, RNA's is uracil
    BASE_COLORS = RNA_BASE_COLORS

    BACKBONE_RADIUS = DNAModifier.BACKBONE_RADIUS
    BACKBONE_SPHERE_RADIUS = DNAModifier.BACKBONE_SPHERE_RADIUS
    BASE_BOND_RADIUS = DNAModifier.BASE_BOND_RADIUS
    BASE_ATOM_RADIUS = DNAModifier.BASE_ATOM_RADIUS

    def __init__(self, n=16, scale=12.0, progress=275, point_count=2000,
                 bases_per_circle=17, axis_length=120.0, shift=(0.0, 0.0, 0.0),
                 plane="xy", backbone_scale=1.1, tube_fraction=0.5,
                 base_colors=None, strand_color="gray_4", strand_scale=1,
                 molecule_color="gray_7", seed=4, name="RNALogo", **kwargs):
        self.n = n
        self.scale = scale
        self.progress = progress
        self.point_count = point_count
        self.bases_per_circle = bases_per_circle
        self.axis_length = axis_length
        self.shift = tuple(shift)
        self.plane = plane
        self.backbone_scale = backbone_scale
        self.tube_fraction = tube_fraction
        self.base_colors = tuple(base_colors or self.BASE_COLORS)
        self.strand_color = strand_color
        self.strand_scale = strand_scale
        self.molecule_color = molecule_color
        self.seed = seed
        self.kwargs = kwargs

        #: the arc length of one lap, and how many bases it holds - the unit
        #: ``Progress`` counts in. Both are closed forms rather than anything
        #: the graph tells us, so that a scene can ramp ``Progress`` to a whole
        #: lap without evaluating the modifier first.
        self.track_length = logo_track_length(n, scale)
        self.track_bases = logo_track_bases(n, bases_per_circle)

        super().__init__(name=name, automatic_layout=False)

    # ------------------------------------------------------------------
    def bases_for_whole_logo(self):
        """How many bases a whole lap of the logo takes."""
        return self.track_bases

    # ------------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        control = self._create_control_frame(tree)
        logo = self._create_logo_frame(tree, control)
        reach = self._create_distance_frame(tree, control, logo)
        ruler = self._create_point_selection_frame(tree, control, logo, reach)
        path = self._create_grow_frame(tree, control, logo, ruler)
        strand = self._create_strand_frame(tree, control, path)
        # the one number that sizes the whole molecule. It sits in no frame
        # because both of the last two want it: the bases are as big as the
        # tube they grow out of, so a circle of a tenth the radius is drawn as
        # the same molecule a tenth the size rather than as a thin wire with
        # full sized bases hanging off it.

        bases = self._create_bases_frame(tree, control, strand)
        backbone = self._create_strand_geometry_frame(tree, control, strand)

        join = JoinGeometry(tree, location=(44.5, 7.4), node_height=GRID,
                            name="JoinMolecule")
        for piece in (backbone, bases):
            tree.links.new(piece, join.geometry_in)
        tree.links.new(join.geometry_out, self.group_outputs.inputs["Geometry"])
        self.group_outputs.location = (46.4 * GRID, 6.8 * GRID)

    # ------------------------------------------------------------------
    def _create_control_frame(self, tree):
        """``ControlFrame``: the five numbers the logo is made of.

        ``Progress`` is the only one a scene touches. The other four say what
        is being drawn rather than how much of it: how many circles, how big,
        how finely the outline is sampled and how many bases each circle
        carries.

        :return: ``{name: node}``, keyed by the label the node carries.
        """
        at = _in_frame((-11.3, 7.0))
        control = {
            "Progress": InputValue(tree, location=at(0.1, -0.1),
                                   value=self.progress, name="Progress",
                                   label="Progress", node_height=GRID),
            "PointCount": InputInteger(tree, location=at(0.1, -0.5),
                                       integer=self.point_count,
                                       name="PointCount", label="PointCount",
                                       node_height=GRID),
            "BasesPerCircle": InputInteger(tree, location=at(0.1, -0.8),
                                           integer=self.bases_per_circle,
                                           name="BasesPerCircle",
                                           label="BasesPerCircle",
                                           node_height=GRID),
            "N": InputInteger(tree, location=at(0.1, -1.1), integer=self.n,
                              name="N", label="N", node_height=GRID),
            "Scale": InputValue(tree, location=at(0.1, -1.5), value=self.scale,
                                name="Scale", label="Scale", node_height=GRID),
            "StrandScale": InputValue(tree, location=at(0.1, -2.1), value=self.strand_scale, name="StrandScale",
                                      label="StrandScale", node_height=GRID),
        }

        palette = [("Base%d" % index, color)
                   for index, color in enumerate(self.base_colors)]
        palette += [("Strand", self.strand_color),
                    ("Molecule", self.molecule_color)]
        for row, (node_name, color) in enumerate(palette):
            control[node_name] = InputMaterial(
                tree, location=at(0.1, -2.4 - 0.5 * row), material=color,
                name=node_name, node_height=GRID, **self.kwargs)
            self.materials.append(control[node_name].node.material)
        control["Palette"] = CombineBundle(
            tree, location=at(1.1, -2.9), name="Palette", node_height=GRID,
            items=[(node_name, "MATERIAL", control[node_name].std_out)
                   for node_name, _ in palette])

        # Progress leaves the frame over a reroute, because two frames a long
        # way apart both want it
        control["ProgressOut"] = Reroute(tree, location=at(1.5, -0.3),
                                         node_height=GRID,
                                         ins=control["Progress"].std_out,
                                         name="ProgressOut")

        frame = Frame(tree, location=(-11.3, 7.0), label="ControlFrame",
                      node_height=GRID)
        frame.add(list(control.values()))
        return control

    # ------------------------------------------------------------------
    def _create_logo_frame(self, tree, control):
        """``CreateLogo Intrinsic Resolution``: the logo, drawn.

        The outline is :func:`objects.logo.logo_curve` evaluated as a field over
        ``PointCount`` points, which is what makes ``N`` a dial rather than a
        decision taken at build time.

        **The formula.** With ``t`` over ``[-pi, pi]`` and ``u = N t``, write
        ``k = floor(|u| / pi)`` and ``s = (-1)^k``::

            a = s cos|u| / 4 + N/4 - k/2        (the chain, before inverting)
            b = sin(u) / 4  + 3/4
            z = (a + i b) / (a^2 + b^2)         (1 / conj, which is inversion)

        ``a + i b`` is a row of half circles of radius ``1/4`` centred at
        ``c = (N/4 - k/2) + 3i/4``; ``k`` counts which one and ``s`` reflects
        every other one so that the row is traversed in one stroke. The
        inversion turns the row into the logo, and the result is halved so that
        it lands in ``x`` in ``[-0.5, 0.5]`` and ``y`` in ``[0, 1]``.

        **Radius and normal, both exactly.** Inversion takes circles to circles:
        the circle through ``c`` of radius ``1/4`` goes to one of radius
        ``0.25 / D`` centred at ``c / D``, where ``D = |c|^2 - 1/16`` comes to
        ``cx^2 + 1/2``. So the arc a base stands on has a radius that can be
        written down - and so has its centre, which is the whole point of doing
        this here. The vector from the point to that centre is

            ``4 D (c/D - z) = (4 cx - 4 D a/d, 3 - 4 D b/d)``

        and it is a *unit* vector already, because the distance from a point on
        a circle to its centre is the radius and the ``4 D`` cancels it. That
        is the inward normal, exact, in two Math nodes' worth of arithmetic and
        with no neighbouring point involved.

        It replaced a finite difference of the strand's own tangents, which
        needed a Sample Index into the strand, a subtraction, a normalise, a
        clamp for the last base and a sign flip - six nodes to approximate
        something the curve already knows. ``D`` is never zero (it is at least
        a half), so there is no case to guard.

        :return: ``{"geometry": ..., "Point": ..., "Radius": ..., "Normal": ...}``
        """
        at = _in_frame((-2.3, 2.2))
        counter = Index(tree, location=(-3.6, 0.0), node_height=GRID,
                        name="OutlineIndex", label="OutlineIndex")
        # the point count is wanted here and again in DistanceToNextPoint
        count = Reroute(tree, location=(-2.9, 1.4), node_height=GRID,
                        ins=control["PointCount"].std_out, name="PointCountOut")

        dense = Points(tree, location=at(0.1, -0.1), count=count.std_out,
                       node_height=GRID, name="OutlinePoints",
                       label="OutlinePoints")

        (horizontal, vertical, other), _ = LOGO_PLANES[self.plane]
        point = [None] * 3
        point[horizontal] = "0.5,a,d,/,*,Scale,*,%r,+" % (self.shift[horizontal],)
        point[vertical] = "0.5,b,d,/,*,Scale,*,%r,+" % (self.shift[vertical],)
        point[other] = "%r" % (self.shift[other],)
        normal = [None] * 3
        normal[horizontal] = "4,cx,*,4,D,*,a,d,/,*,-"
        normal[vertical] = "3,4,D,*,b,d,/,*,-"
        normal[other] = "0"
        # make_function scales y by 100 where every other node in this file
        # uses 200, so its grid coordinate has to be doubled to land on the row
        # the editor puts it on
        spot = at(0.1, -1.2)
        outline = make_function(
            tree, location=(spot[0], 2.0 * spot[1]), name="LogoCurve",
            hide=False,
            inputs=["Index", "N", "Scale", "PointCount"],
            outputs=["Point", "Radius", "Normal"],
            # Index and N are integers *only* - a name listed in both scalars
            # and integers gets two group sockets of the same name
            integers=["Index", "N"], vectors=["Point", "Normal"],
            scalars=["Scale", "Radius", "t", "u", "au", "k", "s", "cx", "a",
                     "b", "d", "D", "PointCount"],
            aux_functions={
                "t": "Index,2,*,pi,*,PointCount,/,pi,-",
                "u": "N,t,*",
                "au": "u,abs",
                "k": "au,%r,/,floor" % (pi,),
                # (-1)^k without a power: 1 - 2 * (k mod 2)
                "s": "1,2,k,2,%,*,-",
                # the centre of the half circle this point is on, along the row
                "cx": "N,4,/,0.5,k,*,-",
                "a": "0.25,s,*,au,cos,*,cx,+",
                "b": "0.25,u,sin,*,0.75,+",
                "d": "a,a,*,b,b,*,+",
                # |c|^2 - r^2, the factor inversion scales this circle by
                "D": "cx,cx,*,0.5,+",
            },
            functions={
                "Point": point,
                "Radius": "0.125,D,/,Scale,*",
                "Normal": normal,
            })
        tree.links.new(counter.std_out, outline.inputs["Index"])
        tree.links.new(control["N"].std_out, outline.inputs["N"])
        tree.links.new(control["Scale"].std_out, outline.inputs["Scale"])
        tree.links.new(count.std_out, outline.inputs["PointCount"])

        placed = SetPosition(tree, location=at(1.2, -0.3),
                             geometry=dense.geometry_out,
                             position=outline.outputs["Point"],
                             node_height=GRID, name="DrawOutline",
                             label="DrawOutline")
        # the radius has to travel *on* the curve, since what reads it is a
        # Sample Curve evaluating a field on the geometry it is sampling
        carried = StoredNamedAttribute(tree, location=at(2.1, -0.3),
                                       data_type="FLOAT", domain="POINT",
                                       name="LogoRadius",
                                       value=outline.outputs["Radius"],
                                       node_height=GRID, label="CarryRadius")
        tree.links.new(placed.geometry_out, carried.geometry_in)

        frame = Frame(tree, location=(-2.3, 2.2), node_height=GRID,
                      label="CreateLogo Intrinsic Resolution")
        frame.add([dense, outline, placed, carried])
        return {"geometry": carried.geometry_out, "index": counter.std_out,
                "count": count.std_out, "Point": outline.outputs["Point"],
                "Radius": outline.outputs["Radius"],
                "Normal": outline.outputs["Normal"]}

    # ------------------------------------------------------------------
    def _create_distance_frame(self, tree, control, logo):
        """``DistanceToNextPoint``: how far it is to the next sample.

        Sample Index does *not* clamp - asked for one past the end it hands
        back a zero vector, and the chord from the last point of the logo to
        the origin would be worth a nonsense number of bases - so the index is
        held at the last point, whose chord to itself is nothing.

        :return: the float socket of the distance.
        """
        at = _in_frame((0.9, 0.2))
        after = MathNode(tree, location=at(0.1, -0.2), operation="ADD",
                         inputs0=logo["index"], inputs1=1.0, node_height=GRID,
                         name="NextOutline", label="NextOutline")
        last = MathNode(tree, location=at(1.0, -0.1), operation="MINIMUM",
                        inputs0=after.std_out, inputs1=logo["count"],
                        node_height=GRID, name="LastOutline",
                        label="LastOutline")
        here = Position(tree, location=at(1.6, -1.4), node_height=GRID,
                        name="OutlinePosition", label="OutlinePosition")
        ahead = SampleIndex(tree, location=at(2.6, -0.5),
                            data_type="FLOAT_VECTOR", domain="POINT",
                            geometry=logo["geometry"], value=here.std_out,
                            index=last.std_out, node_height=GRID,
                            name="PointAhead", label="PointAhead")
        chord = VectorMath(tree, location=at(3.6, -0.9), operation="SUBTRACT",
                           inputs0=ahead.std_out, inputs1=here.std_out,
                           node_height=GRID, name="OutlineChord",
                           label="OutlineChord")
        segment = VectorMath(tree, location=at(4.5, -0.7), operation="LENGTH",
                             inputs0=chord.std_out, node_height=GRID,
                             name="SegmentLength", label="SegmentLength")

        frame = Frame(tree, location=(0.9, 0.2), label="DistanceToNextPoint",
                      node_height=GRID)
        frame.add([after, last, here, ahead, chord, segment])
        return segment.std_out

    # ------------------------------------------------------------------
    def _create_point_selection_frame(self, tree, control, logo, segment):
        """``PointSelection``: the ruler that spaces the bases by curvature.

        A base is entitled to one ``BasesPerCircle``-th of the way round the
        circle it is standing on, so ``Separation = 2 pi r / BasesPerCircle``
        and this stretch of outline is worth ``SegmentLength / Separation``
        bases. Every circle then carries the same number however small it is,
        which is the whole point: a fixed step gives the smallest circles of a
        sixteen-link chain about one base each.

        The running total is the one thing a field cannot do for itself, and
        ``Accumulate Field`` is the node that can - ``Trailing``, which is the
        sum *before* this point, so the first sits at 0 and the last at the
        total. Laying that out along the x axis makes the ``Ruler``: a curve
        whose arc length is the base number, so one Sample Curve by length
        undoes the sum. The logo never becomes a curve at all; it rides along
        as the value being sampled.

        :return: dict with the ruler and how many bases a lap is.
        """
        at = _in_frame((4.1, 3.2))
        radius = NamedAttribute(tree, location=at(0.1, -1.3), data_type="FLOAT",
                                name="LogoRadius", node_height=GRID,
                                label="LogoRadius")
        turn = MathNode(tree, location=at(1.0, -0.8), operation="MULTIPLY",
                        inputs0=radius.std_out, inputs1=2.0 * pi,
                        node_height=GRID, name="Circumference",
                        label="Circumference")
        apart = MathNode(tree, location=at(1.7, -0.3), operation="DIVIDE",
                         inputs0=turn.std_out,
                         inputs1=control["BasesPerCircle"].std_out,
                         node_height=GRID, name="Separation",
                         label="Separation")
        worth = MathNode(tree, location=at(2.8, -0.1), operation="DIVIDE",
                         inputs0=segment, inputs1=apart.std_out,
                         node_height=GRID, name="NecessarySteps",
                         label="NecessarySteps")
        walked = AccumulateField(tree, location=at(4.0, -0.1), data_type="FLOAT",
                                 domain="POINT", value=worth.std_out,
                                 node_height=GRID, name="WalkStations",
                                 label="WalkStations")
        ruler_point = CombineXYZ(tree, location=at(5.2, -0.8), node_height=GRID,
                                 x=walked.trailing, name="RulerPoint",
                                 label="RulerPoint")
        stretched = SetPosition(tree, location=at(6.1, -0.6),
                                geometry=logo["geometry"],
                                position=ruler_point.std_out, node_height=GRID,
                                name="PlaceRuler", label="PlaceRuler")
        ruler = PointsToCurve(tree, location=at(7.1, -0.7), node_height=GRID,
                              name="Ruler", label="Ruler")
        tree.links.new(stretched.geometry_out, ruler.geometry_in)
        tree.links.new(logo["index"], ruler.node.inputs["Weight"])
        # how long a lap is, off the ruler's own length. The accumulator has a
        # Total output that is the same number and cannot be used: it is a
        # *field*, so read from a later frame it would re-run the sum over that
        # frame's geometry. A Curve Length is one value for the whole curve.
        stations = CurveLength(tree, location=at(8.1, -0.9),
                               curve=ruler.geometry_out, node_height=GRID,
                               name="TotalStations", label="TotalStations")

        frame = Frame(tree, location=(4.1, 3.2), label="PointSelection",
                      node_height=GRID)
        frame.add([radius, turn, apart, worth, walked, ruler_point, stretched,
                   ruler, stations])
        return {"ruler": ruler.geometry_out, "radius": radius.std_out,
                "stations": stations.std_out}

    # ------------------------------------------------------------------
    def _create_grow_frame(self, tree, control, logo, ruler):
        """``GrowTheCurve``: which station each base is on, and what is there.

        Base *i* sits at ``Progress - i``, wrapped into ``[0, TotalStations)``.
        The wrap is what a closed track buys: the strand can be driven forwards
        for ever and it laps the logo instead of piling up on the last point,
        and the seam is invisible because the outline's last point is its first.

        Three samples off the ruler at that one length - position, radius and
        normal - and the molecule knows everything it needs.

        :return: ``{"position": ..., "Radius": ..., "Normal": ...}`` per base.
        """
        at = _in_frame((10.8, 1.4))
        carried = Reroute(tree, location=at(4.0, -1.7), node_height=GRID,
                          ins=ruler["ruler"], name="RulerIn")
        index = Index(tree, location=at(0.1, -3.0), node_height=GRID,
                      name="BaseIndex", label="BaseIndex")
        arc = MathNode(tree, location=at(1.1, -2.4), operation="SUBTRACT",
                       inputs0=control["ProgressOut"].std_out,
                       inputs1=index.std_out, node_height=GRID,
                       name="ArcLength", label="ArcLength")
        # Value, Max, Min - and Min has to be written out. Every socket of a
        # Math node defaults to 0.5, so leaving it alone would wrap the track
        # into [0.5, TotalStations) and lose the half station the seam is on.
        lap = MathNode(tree, location=at(3.0, -1.8), operation="WRAP",
                       inputs0=arc.std_out, inputs1=ruler["stations"],
                       node_height=GRID, name="LapHere", label="LapHere")
        lap.node.inputs[2].default_value = 0.0

        samples = {}
        for name, label, kind, value, spot in [
            ("position", "SampleHere", "FLOAT_VECTOR", logo["Point"],
             (4.2, -0.1)),
            ("Radius", "SampleRadius", "FLOAT", ruler["radius"],
             (4.3, -1.9)),
            ("Normal", "SampleNormal", "FLOAT_VECTOR", logo["Normal"],
             (4.3, -3.5))]:
            node = SampleCurve(tree, location=at(*spot), mode="LENGTH",
                               data_type=kind, all_curves=True,
                               node_height=GRID, name=label, label=label)
            tree.links.new(carried.geometry_out, node.geometry_in)
            tree.links.new(lap.std_out, node.node.inputs["Length"])
            tree.links.new(value, node.node.inputs["Value"])
            samples[name] = node

        frame = Frame(tree, location=(10.8, 1.4), label="GrowTheCurve",
                      node_height=GRID)
        frame.add([carried, index, arc, lap] + list(samples.values()))
        return {"position": samples["position"].value_out,
                "Radius": samples["Radius"].value_out,
                "Normal": samples["Normal"].value_out}

    # ------------------------------------------------------------------
    def _create_strand_frame(self, tree, control, path):
        """``Strand``: ``Progress`` points, moved onto the logo.

        The line exists only to carry the points in order - its own length and
        direction are thrown away by the Set Position. How many there are is
        ``Progress`` itself, so the strand gains a base for every station the
        head passes and its tail never moves: one ramp draws the logo.

        The radius and the normal are captured here rather than read again
        downstream, so that the bases and the backbone are certain to be using
        the same numbers as each other.

        :return: dict with the geometry socket and the captured fields.
        """
        at = _in_frame((12.4, 7.7))
        # a strand of no bases is not a strand, and Resample Curve will not
        # make a curve of nought points either

        line = CurveLine(tree, location=at(2.7, -0.2), mode="DIRECTION",
                         direction=[1.0, 0.0, 0.0], length=self.axis_length,
                         node_height=GRID, name="Axis", label="Axis")
        resample = ResampleCurve(tree, location=at(4.2, -0.2),
                                 curve=line.geometry_out, mode="Count",
                                 count=control["ProgressOut"].std_out, node_height=GRID,
                                 name="OnePointPerBase",
                                 label="OnePointPerBase")
        axis = SetPosition(tree, location=at(5.7, -0.2),
                           geometry=resample.geometry_out,
                           position=path["position"], node_height=GRID,
                           name="OntoTrack", label="OntoTrack")

        index = Index(tree, location=at(5.7, -2.3), node_height=GRID,
                      name="TurnIndex", label="TurnIndex")
        capture = CaptureAttribute(
            tree, location=at(6.7, -0.1), domain="POINT",
            geometry=axis.geometry_out,
            items=[("Normal", "FLOAT_VECTOR", path["Normal"]),
                   ("Radius", "FLOAT", path["Radius"]),
                   ("BaseIndex", "INT", index.std_out)],
            node_height=GRID, name="CaptureSpoke", label="CaptureSpoke")

        frame = Frame(tree, location=(12.4, 7.7), label="Strand",
                      node_height=GRID)
        frame.add([line, resample, axis,
                   index, capture])
        return {
            "geometry": capture.geometry_out,
            "Normal": capture["Normal"],
            "Radius": capture["Radius"],
            "BaseIndex": capture["BaseIndex"],
        }

    # ------------------------------------------------------------------

    def _create_bases_frame(self, tree, control, strand):
        """``Bases``: one base per point of the backbone, pointing inwards.

        A base is a short chain of 2..5 atoms - spheres on a mesh line, with
        the line itself swept to a tube for the bonds. How many atoms is what
        distinguishes the four base types on screen.

        Which type a base is comes from a random draw with the base's own index
        as its id, not from its position: a strand that grows along the logo has
        to keep the sequence it was dealt, and a sequence read off the curve
        would be rewritten under the molecule as it moved.

        The type and the size go into the zone as input items rather than being
        read inside it, and that is not a matter of taste. Inside the zone the
        type has to drive two Index Switches, and neither the number of atoms a
        Mesh Line is built with nor a material is a field; a Named Attribute
        read in here would have no geometry to be evaluated on and both
        switches would quietly fall back to their first slot - one base type,
        drawn four times in the same colour. A field has to be evaluated where
        its geometry is, which is out here, once per element.

        :param size: how big a base is - the same number the backbone is drawn
            at, so that the molecule keeps its proportions on every circle.
        :return: the geometry socket of all the bases.
        """
        at = _in_frame((27.1, 10.2))
        draw = RandomValue(tree, location=at(0.1, -2.2), data_type="INT",
                           min=0, max=len(self.BASE_ATOMS) - 1, seed=self.seed,
                           node_height=GRID, name="WhichBase",
                           label="WhichBase")
        tree.links.new(strand["BaseIndex"], draw.node.inputs["ID"])

        # the base's chain is built along +z and Align Rotation to Vector turns
        # that onto the normal, which points at the centre of the circle this
        # base is standing on
        aim = AlignRotationToVector(tree, location=at(0.1, -3.7), axis="Z",
                                    pivot_axis="AUTO", vector=strand["Normal"],
                                    node_height=GRID, name="AimInwards",
                                    label="AimInwards")

        half_radius = MathNode(tree, operation="MULTIPLY", loaction=at(0.1, -4.7), inputs0=strand["Radius"],
                               inputs1=0.5,
                               label="HalfRadius")

        zone = ForEachZone(tree, location=at(3.1, -0.2), domain="POINT",
                           node_width=10.5, geometry=strand["geometry"],
                           node_height=GRID, name="ForEachBase",
                           label="ForEachBase")
        zone.add_socket("INT", "BaseType", value=draw.std_out, for_input=True)
        zone.add_socket("ROTATION", "Rotation", value=aim.std_out,
                        for_input=True)
        zone.add_socket("FLOAT", "BaseSize", value=half_radius.std_out, for_input=True)
        zone.foreach_output.location = tuple(v * GRID for v in at(13.6, -0.2))
        base_type = zone.foreach_input.outputs["BaseType"]

        atoms = IndexSwitch(tree, location=at(6.1, -2.2), data_type="INT",
                            index=base_type, node_height=GRID,
                            name="AtomsPerBase", label="AtomsPerBase")
        for _ in range(len(self.BASE_ATOMS) - 2):
            atoms.new_item()
        for slot, number in enumerate(self.BASE_ATOMS):
            atoms.node.inputs[slot + 1].default_value = number

        chain = MeshLine(tree, location=at(7.6, -2.2), mode="END_POINTS",
                         count_mode="TOTAL", count=atoms.std_out,
                         start_location=[0.0, 0.0, 0.0], node_height=GRID,
                         name="BaseChain", label="BaseChain")
        chain.node.inputs["Offset"].default_value = [0.0, 0.0, 1.0]
        chain_out = Reroute(tree, location=at(8.6, -2.2), node_height=GRID,
                            ins=chain.geometry_out, name="ChainOut",
                            label="ChainOut")

        atom = IcoSphere(tree, location=at(7.0, -0.7),
                         radius=self.BASE_ATOM_RADIUS, subdivisions=2,
                         node_height=GRID, name="Atom", label="Atom")
        atom_instances = InstanceOnPoints(tree, location=at(9.6, -0.7),
                                          points=chain_out.geometry_out,
                                          instance=atom.geometry_out,
                                          node_height=GRID, name="Atoms",
                                          label="Atoms")

        bonds = MeshToCurve(tree, location=at(9.6, -2.4),
                            mesh=chain_out.geometry_out, node_height=GRID,
                            name="Bonds", label="Bonds")
        bond_profile = CurveCircle(tree, location=at(7.0, -3.7), mode="RADIUS",
                                   resolution=16, radius=self.BASE_BOND_RADIUS,
                                   node_height=GRID, name="BondProfile",
                                   label="BondProfile")
        bond_mesh = CurveToMesh(tree, location=at(11.1, -2.2),
                                curve=bonds.geometry_out,
                                profile_curve=bond_profile.geometry_out,
                                fill_caps=False, node_height=GRID,
                                name="BondMesh", label="BondMesh")

        base = JoinGeometry(tree, location=at(12.1, -1.2), node_height=GRID,
                            name="JoinBase", label="JoinBase")
        for piece in (atom_instances.geometry_out, bond_mesh.geometry_out):
            tree.links.new(piece, base.geometry_in)

        # a material cannot be picked by a selection the way geometry can - Set
        # Material takes one - so here the choice really is a switch
        colors = SeparateBundle(
            tree, location=at(4.0, -4.7), bundle=control["Palette"].std_out,
            items=[("Base%d" % index, "MATERIAL")
                   for index in range(len(self.BASE_ATOMS))],
            node_height=GRID, name="BaseColors", label="BaseColors")
        material = IndexSwitch(tree, location=at(6.1, -4.7),
                               data_type="MATERIAL", index=base_type,
                               node_height=GRID, name="BaseMaterial",
                               label="BaseMaterial")
        for index in range(len(self.BASE_ATOMS)):
            material.add_item(socket=colors.out("Base%d" % index))

        placed = InstanceOnPoints(tree, location=at(13.1, -0.2),
                                  points=zone.element,
                                  instance=base.geometry_out,
                                  rotation=zone.foreach_input.outputs["Rotation"],
                                  scale=zone.foreach_input.outputs["BaseSize"],
                                  node_height=GRID, name="PlaceBase",
                                  label="PlaceBase")
        paint = SetMaterial(tree, location=at(14.1, -0.2),
                            geometry=placed.geometry_out,
                            material=material.std_out, node_height=GRID,
                            name="PaintBase", label="PaintBase")
        tree.links.new(paint.geometry_out,
                       zone.foreach_output.inputs["Geometry"])

        frame = Frame(tree, location=(27.1, 10.2), label="Bases",
                      node_height=GRID)
        frame.add([draw, aim, zone, atoms, chain, chain_out, atom,
                   atom_instances, bonds, bond_profile, bond_mesh, base,
                   colors, material, placed, paint, half_radius])
        return zone.geometry_out

    # ------------------------------------------------------------------
    def _create_strand_geometry_frame(self, tree, control, strand):
        """``Strand Geometry``: the backbone as a solid tube.

        The curve is swept to a tube for the sugar-phosphate backbone, and a
        sphere is dropped on every point so that it reads as a chain of atoms
        rather than a smooth pipe. Both are scaled by the local radius, so the
        molecule thins with the circle it is wound round instead of swallowing
        the bases where the chain gets small.

        :param bead: scale of the spheres.
        :param tube: scale of the swept profile.
        :return: the geometry socket of the backbone.
        """
        at = _in_frame((26.3, 3.0))
        curve = Reroute(tree, location=at(0.1, -1.1), node_height=GRID,
                        ins=strand["geometry"], name="StrandIn",
                        label="StrandIn")

        strand_scale = MathNode(tree, operation="MULTIPLY", location=at(0.1, -4.7),
                                inputs0=control["StrandScale"].std_out,
                                inputs1=strand["Radius"], label="StrandScale")

        colors = SeparateBundle(
            tree, location=at(1.0, -1.1), bundle=control["Palette"].std_out,
            items=[("Strand", "MATERIAL"), ("Molecule", "MATERIAL")],
            node_height=GRID, name="BackboneColors", label="BackboneColors")

        profile = CurveCircle(tree, location=at(0.8, -2.5), mode="RADIUS",
                              resolution=16, radius=self.BACKBONE_RADIUS,
                              node_height=GRID, name="BackboneProfile",
                              label="BackboneProfile")
        pipe = CurveToMesh(tree, location=at(3.5, -2.7),
                           curve=curve.geometry_out,
                           profile_curve=profile.geometry_out,
                           fill_caps=False, node_height=GRID, name="Backbone",
                           label="Backbone")
        tree.links.new(strand_scale.std_out, pipe.node.inputs["Scale"])
        pipe_material = SetMaterial(tree, location=at(4.6, -2.6),
                                    geometry=pipe.geometry_out,
                                    material=colors.out("Strand"),
                                    node_height=GRID, name="PaintBackbone",
                                    label="PaintBackbone")

        atom = UVSphere(tree, location=at(1.0, -0.1),
                        radius=self.BACKBONE_SPHERE_RADIUS, segments=16,
                        rings=8, node_height=GRID, name="BackboneAtom",
                        label="BackboneAtom")
        atoms = InstanceOnPoints(tree, location=at(3.1, -0.1),
                                 points=curve.geometry_out,
                                 instance=atom.geometry_out, scale=strand_scale.std_out,
                                 node_height=GRID, name="BackboneAtoms",
                                 label="BackboneAtoms")
        atom_material = SetMaterial(tree, location=at(4.6, -0.1),
                                    geometry=atoms.geometry_out,
                                    material=colors.out("Molecule"),
                                    node_height=GRID, name="PaintAtoms",
                                    label="PaintAtoms")

        join = JoinGeometry(tree, location=at(6.1, -1.1), node_height=GRID,
                            name="JoinBackbone", label="JoinBackbone")
        for piece in (pipe_material.geometry_out, atom_material.geometry_out):
            tree.links.new(piece, join.geometry_in)
        smooth = SetShadeSmooth(tree, location=at(7.6, -1.1),
                                geometry=join.geometry_out, node_height=GRID,
                                name="SmoothBackbone", label="SmoothBackbone")

        frame = Frame(tree, location=(26.3, 3.0), label="Strand Geometry",
                      node_height=GRID)
        frame.add([curve, colors, profile, pipe, pipe_material, atom, atoms,
                   atom_material, join, smooth, strand_scale])
        return smooth.geometry_out


class RNACircleModifier(GeometryNodesModifier):
    """A single strand of RNA wound once round a circle.

    :class:`RNALogoModifier` grows a strand along the logo; this grows the same
    molecule along the simplest closed track there is. The two share everything
    downstream of the backbone - the ``Bases`` and ``Strand Geometry`` frames
    are the same nodes wired the same way - and differ only in how a point of
    the backbone learns where it is, which way it faces and how sharply the
    track turns under it.

    On the logo those three came out of a ``Sample Curve`` on a ruler built by
    an ``Accumulate Field``, because the track's curvature varies and a base has
    to be given the share of arc length its own circle is entitled to. Here the
    curvature is one number, so all three collapse to arithmetic:

    ``Position``
        base *i* sits at ``Radius * (cos t, 0, sin t)`` with
        ``t = 2 pi i / (BasesPerCircle - 1)`` - a circle in the x-z plane, the
        plane the flat scenes of this video are built in.
    ``Radius``
        ``Radius``. The radius of curvature of a circle is the circle's radius,
        so the number that sizes the molecule on the logo is a constant here -
        the wiring is :class:`RNALogoModifier`'s, and it still means what it
        says.
    ``Normal``
        ``cross(tangent, y)``, which on this circle comes to ``-(cos t, 0, sin t)``:
        the inward radius, exactly, and already a unit vector. It is scaled by
        ``Radius`` before it is captured, which changes nothing - Align Rotation
        to Vector only reads the direction - and keeps it the same length as the
        radius it stands for.

    Three frames, then, rather than seven:

    ``ControlFrame``
        Four numbers and the palette. ``Progress`` is the whole of the
        choreography, as on the logo: it is how many bases have been laid down.
    ``Strand``
        The circle, and how much of it exists. A line resampled to
        ``BasesPerCircle`` points, moved onto the circle by the arithmetic
        above, cut down to the points before the head by a Separate Geometry,
        and the radius, the normal and the base number captured onto what is
        left.
    ``Bases`` and ``Strand Geometry``
        A base per point, aimed down the normal, and the backbone swept to a
        tube - :class:`RNALogoModifier`'s two frames, unchanged.

    **The seam.** The angle steps by ``2 pi / (BasesPerCircle - 1)``, so the
    last point of the strand lands exactly on the first: a full lap that closes
    to the last bit, at the price of one base drawn twice where the ends meet.
    That is what makes ``Progress`` safe to ramp all the way to
    ``BasesPerCircle`` - the strand arrives back where it started rather than
    stopping short of itself.

    :param progress: how many bases have been laid down, counting from the one
        at angle zero. Animate this and the ring draws itself. Points from
        ``Progress`` on are separated away rather than scaled to nothing, so
        the ones that are not there yet cost nothing.
    :param bases_per_circle: bases in a whole lap - both the number of points
        the backbone is resampled to and the number the angle is divided into.
    :param scale: the radius of the circle, and, through ``Radius``, the size
        of the molecule riding on it.
    :param strand_scale: backbone thickness per unit of radius. The swept tube
        and the beads on it are both drawn at ``strand_scale * Radius``.
    :param base_colors: the four base materials, in ``BaseType`` order.
    :param strand_color: material of the swept backbone.
    :param molecule_color: material of the spheres sitting on the backbone.
    :param seed: which strand of RNA this is. Changing it deals the bases again.
    """

    #: the two purines and the two pyrimidines, drawn with as many atoms as
    #: :class:`DNAModifier` draws them with
    BASE_ATOMS = DNAModifier.BASE_ATOMS
    BASE_NAMES = RNALogoModifier.BASE_NAMES
    #: three of DNA's four base colours and one that is not - DNA's fourth base
    #: is thymine, RNA's is uracil
    BASE_COLORS = RNALogoModifier.BASE_COLORS

    BACKBONE_RADIUS = DNAModifier.BACKBONE_RADIUS
    BACKBONE_SPHERE_RADIUS = DNAModifier.BACKBONE_SPHERE_RADIUS
    BASE_BOND_RADIUS = DNAModifier.BASE_BOND_RADIUS
    BASE_ATOM_RADIUS = DNAModifier.BASE_ATOM_RADIUS

    def __init__(self, progress=18, bases_per_circle=18, radius=1.0, scale=1,
                 strand_scale=1.0, base_colors=None, strand_color="gray_4",
                 molecule_color="gray_7", seed=4, name="RNACircle", **kwargs):
        self.progress = progress
        self.bases_per_circle = bases_per_circle
        self.scale = scale
        self.radius = radius
        self.strand_scale = strand_scale
        self.base_colors = tuple(base_colors or self.BASE_COLORS)
        self.strand_color = strand_color
        self.molecule_color = molecule_color
        self.seed = seed
        self.kwargs = kwargs

        super().__init__(name=name, automatic_layout=False)

    # ------------------------------------------------------------------
    def bases_for_whole_circle(self):
        """How many bases a whole lap takes - what to ramp ``Progress`` to.

        The counterpart of :meth:`RNALogoModifier.bases_for_whole_logo`, and
        here it is simply the dial itself: the circle is divided into
        ``BasesPerCircle`` stations and there is exactly one base on each.
        """
        return self.bases_per_circle

    # ------------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        control = self._create_control_frame(tree)
        strand = self._create_strand_frame(tree, control)
        bases = self._create_bases_frame(tree, control, strand)
        backbone = self._create_strand_geometry_frame(tree, control, strand)

        join = JoinGeometry(tree, location=(45.3, 7.4), node_height=GRID,
                            name="JoinMolecule")
        for piece in (backbone, bases):
            tree.links.new(piece, join.geometry_in)

        transform_geometry = TransformGeometry(tree, location=(46.3, 7.4), translation=control["Translation"].std_out,
                                               rotation=control["Rotation"].std_out, name="FinalTransform")
        create_geometry_line(tree, [join, transform_geometry], out=self.group_outputs.inputs["Geometry"])

        self.group_outputs.location = (47.2 * GRID, 6.8 * GRID)

    # ------------------------------------------------------------------
    def _create_control_frame(self, tree):
        """``ControlFrame``: the four numbers the ring is made of.

        ``Progress`` is the only one a scene touches. ``BasesPerCircle`` says
        how finely the lap is divided, ``Radius`` how big it is, and
        ``StrandScale`` how fat the backbone is drawn.

        :return: ``{name: node}``, keyed by the label the node carries.
        """
        at = _in_frame((10.5, 3.7))
        control = {
            "Translation": InputVector(tree, location=at(0.1, 0.5), name="Translation", label="Translation", hide=True,
                                       vector=Vector()),
            "Rotation": InputRotation(tree, location=at(0.1, 1.1), name="Rotation", label="Rotation", hide=True,
                                      rotation=Vector()),
            "Scale": InputValue(tree, location=at(0.1, 1.6), name="Scale", label="Scale", hide=True, value=self.scale),
            "Progress": InputInteger(tree, location=at(0.1, -0.1),
                                     integer=self.progress, name="Progress",
                                     label="Progress", node_height=GRID),
            "BasesPerCircle": InputInteger(tree, location=at(0.1, -0.5),
                                           integer=self.bases_per_circle,
                                           name="BasesPerCircle",
                                           label="BasesPerCircle",
                                           node_height=GRID),
            "Radius": InputValue(tree, location=at(0.1, -0.8), value=self.radius,
                                 name="Radius", label="Radius", node_height=GRID),
            "StrandScale": InputValue(tree, location=at(0.1, -1.1),
                                      value=self.strand_scale,
                                      name="StrandScale", label="StrandScale",
                                      node_height=GRID),
        }

        # the multiplier of a SCALE goes in through ``float_input``: it is the
        # node's fourth socket, ``Scale``, not its second ``Vector`` - and
        # ``inputs1`` writes that second Vector, which SCALE disables and the
        # editor therefore does not even draw
        trans_scale = VectorMath(tree, location=at(1.1, 0.5), operation="SCALE", inputs0=control["Translation"].std_out,
                                 float_input=control["Scale"].std_out, hide=True)

        palette = [("Base%d" % index, color)
                   for index, color in enumerate(self.base_colors)]
        palette += [("Strand", self.strand_color),
                    ("Molecule", self.molecule_color)]
        for row, (node_name, color) in enumerate(palette):
            control[node_name] = InputMaterial(
                tree, location=at(0.1, -2.4 - 0.5 * row), material=color,
                name=node_name, node_height=GRID, **self.kwargs)
            self.materials.append(control[node_name].node.material)
        control["Palette"] = CombineBundle(
            tree, location=at(1.1, -2.9), name="Palette", node_height=GRID,
            items=[(node_name, "MATERIAL", control[node_name].std_out)
                   for node_name, _ in palette])

        frame = Frame(tree, location=(10.5, 3.7), label="ControlFrame",
                      node_height=GRID)
        frame.add(list(control.values()) + [trans_scale])
        control["Translation"] = trans_scale
        return control

    # ------------------------------------------------------------------
    def _create_strand_frame(self, tree, control):
        """``Strand``: the circle, and how much of it there is so far.

        A ``Curve Line`` resampled to ``BasesPerCircle`` points - the line's own
        endpoints are thrown away by the Set Position and only its point order
        survives - laid onto ``Radius * (cos t, 0, sin t)``. The two chains of
        Math nodes that build ``t`` are the same expression twice, once for the
        cosine and once for the sine, since a Math node has one output.

        The four nodes that count the stations sit *outside* the frame: the
        divisor and the index are read by both chains, and there is nothing
        about them that belongs to one side of the circle.

        Separate Geometry then keeps the points with ``Index < Progress``.
        Building the head this way rather than by scaling the tail to nothing
        means the bases beyond it are never built, and it is what lets the last
        node of the frame capture onto the strand that actually exists.

        The capture is the reason this frame ends where it does. Downstream the
        backbone becomes a tube and the bases become instances - two topology
        changes - and a field read after either of them would be evaluated on
        the wrong geometry, so the normal, the radius and the base's number are
        frozen onto the points here, once.

        :return: dict with the geometry socket and the three captured fields.
        """
        at = _in_frame((15.6, 8.6))

        # how many steps a lap is divided into. One less than the number of
        # points, so that the last point lands on the first
        divisions = MathNode(tree, location=(12.5, 6.3), operation="SUBTRACT",
                             inputs0=control["BasesPerCircle"].std_out,
                             inputs1=1.0, node_height=GRID, name="Divisions",
                             label="Divisions")
        station = Index(tree, location=(14.6, 6.3), node_height=GRID,
                        name="CircleIndex", label="CircleIndex")

        line = CurveLine(tree, location=(14.8, 10.4), mode="POINTS",
                         start=Vector([0.0, 0.0, 0.0]),
                         end=Vector([0.0, 0.0, 1.0]), node_height=GRID,
                         name="Axis", label="Axis")
        resample = ResampleCurve(tree, location=(16.0, 10.1), mode="Count",
                                 curve=line.geometry_out,
                                 count=control["BasesPerCircle"].std_out,
                                 node_height=GRID, name="OnePointPerBase",
                                 label="OnePointPerBase")

        # ``Radius`` is read four times in this frame. The two multiplies that
        # size the circle take it over a reroute; the normal and the capture
        # read the dial itself
        radius = Reroute(tree, location=at(2.8, -2.5), node_height=GRID,
                         ins=control["Radius"].std_out, name="RadiusOut",
                         label="RadiusOut")

        # x = Radius * cos(2 pi i / divisions)
        turn_x = MathNode(tree, location=at(0.1, -1.2), operation="MULTIPLY",
                          inputs0=station.std_out, inputs1=2.0 * pi,
                          node_height=GRID, name="FullTurnX", label="FullTurnX")
        angle_x = MathNode(tree, location=at(1.0, -1.1), operation="DIVIDE",
                           inputs0=turn_x.std_out, inputs1=divisions.std_out,
                           node_height=GRID, name="AngleX", label="AngleX")
        cosine = MathNode(tree, location=at(1.9, -1.1), operation="COSINE",
                          inputs0=angle_x.std_out, node_height=GRID,
                          name="CosAngle", label="CosAngle")
        circle_x = MathNode(tree, location=at(3.0, -1.0), operation="MULTIPLY",
                            inputs0=cosine.std_out, inputs1=radius.std_out,
                            node_height=GRID, name="CircleX", label="CircleX")

        # z = Radius * sin(2 pi i / divisions), the same expression again
        turn_z = MathNode(tree, location=at(0.1, -1.9), operation="MULTIPLY",
                          inputs0=station.std_out, inputs1=2.0 * pi,
                          node_height=GRID, name="FullTurnZ", label="FullTurnZ")
        angle_z = MathNode(tree, location=at(1.0, -1.8), operation="DIVIDE",
                           inputs0=turn_z.std_out, inputs1=divisions.std_out,
                           node_height=GRID, name="AngleZ", label="AngleZ")
        sine = MathNode(tree, location=at(1.9, -1.9), operation="SINE",
                        inputs0=angle_z.std_out, node_height=GRID,
                        name="SinAngle", label="SinAngle")
        circle_z = MathNode(tree, location=at(3.0, -1.9), operation="MULTIPLY",
                            inputs0=sine.std_out, inputs1=radius.std_out,
                            node_height=GRID, name="CircleZ", label="CircleZ")

        # the circle lies in x-z, facing a camera that looks along +y, like the
        # rest of the flat scenes of this video
        point = CombineXYZ(tree, location=at(4.7, -1.5), x=circle_x.std_out,
                           z=circle_z.std_out, node_height=GRID,
                           name="CirclePoint", label="CirclePoint")
        ring = SetPosition(tree, location=at(5.6, -1.0),
                           geometry=resample.geometry_out,
                           position=point.std_out, node_height=GRID,
                           name="OntoCircle", label="OntoCircle")

        grow_index = Index(tree, location=at(3.5, -3.7), node_height=GRID,
                           name="GrowIndex", label="GrowIndex")
        grown = CompareNode(tree, location=at(5.3, -3.4), data_type="INT",
                            operation="LESS_THAN", inputs0=grow_index.std_out,
                            inputs1=control["Progress"].std_out,
                            node_height=GRID, name="Grown", label="Grown")
        head = SeparateGeometry(tree, location=at(6.8, -2.9), domain="POINT",
                                selection=grown.std_out, node_height=GRID,
                                name="GrowStrand", label="GrowStrand")
        # SeparateGeometry takes a `geometry` keyword but does not wire it
        tree.links.new(ring.geometry_out, head.geometry_in)

        # the inward radius of the circle, which is where a base has to point.
        # Curve Tangent is already a unit vector and so is the cross product of
        # two perpendicular ones, so there is nothing to normalise
        position = Position(tree, location=at(5.7, -0.1), node_height=GRID, label="Radius")

        normal = VectorMath(tree, location=at(7.4, -1.1), operation="SCALE",
                            inputs0=position.std_out,
                            float_input=-1,
                            node_height=GRID, name="InwardNormal",
                            label="InwardNormal")

        index = Index(tree, location=at(6.8, -3.7), node_height=GRID,
                      name="TurnIndex", label="TurnIndex")
        capture = CaptureAttribute(
            tree, location=at(8.9, -1.3), domain="POINT",
            geometry=head.geometry_out,
            items=[("Normal", "FLOAT_VECTOR", normal.std_out),
                   # the radius of curvature of a circle is its own radius
                   ("Radius", "FLOAT", control["Radius"].std_out),
                   ("BaseIndex", "INT", index.std_out)],
            node_height=GRID, name="CaptureSpoke", label="CaptureSpoke")

        frame = Frame(tree, location=(15.6, 8.6), label="Strand",
                      node_height=GRID)
        frame.add([radius, turn_x, angle_x, cosine, circle_x, turn_z, angle_z,
                   sine, circle_z, point, ring, grow_index, grown, head,
                   position, normal, index, capture])
        return {
            "geometry": capture.geometry_out,
            "Normal": capture["Normal"],
            "Radius": capture["Radius"],
            "BaseIndex": capture["BaseIndex"],
        }

    # ------------------------------------------------------------------
    def _create_bases_frame(self, tree, control, strand):
        """``Bases``: one base per point of the backbone, pointing inwards.

        :class:`RNALogoModifier`'s frame, node for node. A base is a short chain
        of 2..5 atoms - spheres on a mesh line, with the line itself swept to a
        tube for the bonds - and how many atoms is what tells the four base
        types apart on screen.

        Which type a base is comes from a random draw with the base's own index
        as its id rather than from where it is, so that a strand that grows
        keeps the sequence it was dealt instead of having it rewritten
        underneath as it moves.

        The type and the size go into the zone as input items rather than being
        read inside it, and that is not a matter of taste. Inside the zone the
        type has to drive two Index Switches, and neither the number of atoms a
        Mesh Line is built with nor a material is a field; a field read in there
        would have no geometry to be evaluated on and both switches would
        quietly fall back to their first slot - one base type, drawn four times
        in the same colour.

        :return: the geometry socket of all the bases.
        """
        at = _in_frame((27.9, 10.2))
        draw = RandomValue(tree, location=at(0.1, -2.2), data_type="INT",
                           min=0, max=len(self.BASE_ATOMS) - 1, seed=self.seed,
                           node_height=GRID, name="WhichBase",
                           label="WhichBase")
        tree.links.new(strand["BaseIndex"], draw.node.inputs["ID"])

        # the base's chain is built along +z and Align Rotation to Vector turns
        # that onto the normal, which points at the centre of the circle
        aim = AlignRotationToVector(tree, location=at(0.1, -3.7), axis="Z",
                                    pivot_axis="AUTO", vector=strand["Normal"],
                                    node_height=GRID, name="AimInwards",
                                    label="AimInwards")

        half_radius = MathNode(tree, location=at(0.2, -4.8),
                               operation="MULTIPLY", inputs0=strand["Radius"],
                               inputs1=0.5, node_height=GRID,
                               name="HalfRadius", label="HalfRadius")

        zone = ForEachZone(tree, location=at(3.1, -0.2), domain="POINT",
                           node_width=10.5, geometry=strand["geometry"],
                           node_height=GRID, name="ForEachBase",
                           label="ForEachBase")
        zone.add_socket("INT", "BaseType", value=draw.std_out, for_input=True)
        zone.add_socket("ROTATION", "Rotation", value=aim.std_out,
                        for_input=True)
        zone.add_socket("FLOAT", "BaseSize", value=half_radius.std_out,
                        for_input=True)
        zone.foreach_output.location = tuple(v * GRID for v in at(13.6, -0.2))
        base_type = zone.foreach_input.outputs["BaseType"]

        atoms = IndexSwitch(tree, location=at(6.1, -2.2), data_type="INT",
                            index=base_type, node_height=GRID,
                            name="AtomsPerBase", label="AtomsPerBase")
        for _ in range(len(self.BASE_ATOMS) - 2):
            atoms.new_item()
        for slot, number in enumerate(self.BASE_ATOMS):
            atoms.node.inputs[slot + 1].default_value = number

        chain = MeshLine(tree, location=at(7.6, -2.2), mode="END_POINTS",
                         count_mode="TOTAL", count=atoms.std_out,
                         start_location=[0.0, 0.0, 0.0], node_height=GRID,
                         name="BaseChain", label="BaseChain")
        chain.node.inputs["Offset"].default_value = [0.0, 0.0, 1.0]
        chain_out = Reroute(tree, location=at(8.6, -2.2), node_height=GRID,
                            ins=chain.geometry_out, name="ChainOut",
                            label="ChainOut")

        atom = IcoSphere(tree, location=at(6.1, -0.7),
                         radius=self.BASE_ATOM_RADIUS, subdivisions=2,
                         node_height=GRID, name="Atom", label="Atom")
        atom_instances = InstanceOnPoints(tree, location=at(9.6, -0.7),
                                          points=chain_out.geometry_out,
                                          instance=atom.geometry_out,
                                          node_height=GRID, name="Atoms",
                                          label="Atoms")

        bonds = MeshToCurve(tree, location=at(9.6, -2.4),
                            mesh=chain_out.geometry_out, node_height=GRID,
                            name="Bonds", label="Bonds")
        bond_profile = CurveCircle(tree, location=at(6.1, -3.7), mode="RADIUS",
                                   resolution=16, radius=self.BASE_BOND_RADIUS,
                                   node_height=GRID, name="BondProfile",
                                   label="BondProfile")
        bond_mesh = CurveToMesh(tree, location=at(11.1, -2.2),
                                curve=bonds.geometry_out,
                                profile_curve=bond_profile.geometry_out,
                                fill_caps=False, node_height=GRID,
                                name="BondMesh", label="BondMesh")

        base = JoinGeometry(tree, location=at(12.1, -1.2), node_height=GRID,
                            name="JoinBase", label="JoinBase")
        for piece in (atom_instances.geometry_out, bond_mesh.geometry_out):
            tree.links.new(piece, base.geometry_in)

        # a material cannot be picked by a selection the way geometry can - Set
        # Material takes one - so here the choice really is a switch
        colors = SeparateBundle(
            tree, location=at(3.1, -4.7), bundle=control["Palette"].std_out,
            items=[("Base%d" % index, "MATERIAL")
                   for index in range(len(self.BASE_ATOMS))],
            node_height=GRID, name="BaseColors", label="BaseColors")
        material = IndexSwitch(tree, location=at(6.1, -4.7),
                               data_type="MATERIAL", index=base_type,
                               node_height=GRID, name="BaseMaterial",
                               label="BaseMaterial")
        for index in range(len(self.BASE_ATOMS)):
            material.add_item(socket=colors.out("Base%d" % index))

        placed = InstanceOnPoints(tree, location=at(13.1, -0.2),
                                  points=zone.element,
                                  instance=base.geometry_out,
                                  rotation=zone.foreach_input.outputs["Rotation"],
                                  scale=zone.foreach_input.outputs["BaseSize"],
                                  node_height=GRID, name="PlaceBase",
                                  label="PlaceBase")
        paint = SetMaterial(tree, location=at(14.1, -0.2),
                            geometry=placed.geometry_out,
                            material=material.std_out, node_height=GRID,
                            name="PaintBase", label="PaintBase")
        tree.links.new(paint.geometry_out,
                       zone.foreach_output.inputs["Geometry"])

        frame = Frame(tree, location=(27.9, 10.2), label="Bases",
                      node_height=GRID)
        frame.add([draw, aim, half_radius, zone, atoms, chain, chain_out, atom,
                   atom_instances, bonds, bond_profile, bond_mesh, base,
                   colors, material, placed, paint])
        return zone.geometry_out

    # ------------------------------------------------------------------
    def _create_strand_geometry_frame(self, tree, control, strand):
        """``Strand Geometry``: the backbone as a solid tube.

        :class:`RNALogoModifier`'s frame again. The curve is swept to a tube for
        the sugar-phosphate backbone, and a sphere is dropped on every point so
        that it reads as a chain of atoms rather than a smooth pipe. Both are
        drawn at ``StrandScale * Radius``, which on a circle is one number for
        the whole strand - the same expression that made the molecule thin with
        the logo's smaller circles simply holds still here.

        :return: the geometry socket of the backbone.
        """
        at = _in_frame((27.1, 3.9))
        curve = Reroute(tree, location=at(0.3, -1.1), node_height=GRID,
                        ins=strand["geometry"], name="StrandIn",
                        label="StrandIn")

        # labelled after the dial it multiplies, but not *named* after it - two
        # nodes of one name is one node called ``StrandScale.001``
        strand_scale = MathNode(tree, location=at(0.3, -3.9),
                                operation="MULTIPLY",
                                inputs0=control["StrandScale"].std_out,
                                inputs1=strand["Radius"], node_height=GRID,
                                name="BackboneScale", label="StrandScale")

        colors = SeparateBundle(
            tree, location=at(0.3, -1.7), bundle=control["Palette"].std_out,
            items=[("Strand", "MATERIAL"), ("Molecule", "MATERIAL")],
            node_height=GRID, name="BackboneColors", label="BackboneColors")

        profile = CurveCircle(tree, location=at(0.1, -2.5), mode="RADIUS",
                              resolution=16, radius=self.BACKBONE_RADIUS,
                              node_height=GRID, name="BackboneProfile",
                              label="BackboneProfile")
        pipe = CurveToMesh(tree, location=at(3.4, -2.4),
                           curve=curve.geometry_out,
                           profile_curve=profile.geometry_out,
                           fill_caps=False, node_height=GRID, name="Backbone",
                           label="Backbone")
        tree.links.new(strand_scale.std_out, pipe.node.inputs["Scale"])
        pipe_material = SetMaterial(tree, location=at(4.8, -2.6),
                                    geometry=pipe.geometry_out,
                                    material=colors.out("Strand"),
                                    node_height=GRID, name="PaintBackbone",
                                    label="PaintBackbone")

        atom = UVSphere(tree, location=at(0.3, -0.1),
                        radius=self.BACKBONE_SPHERE_RADIUS, segments=16,
                        rings=8, node_height=GRID, name="BackboneAtom",
                        label="BackboneAtom")
        atoms = InstanceOnPoints(tree, location=at(3.3, -0.1),
                                 points=curve.geometry_out,
                                 instance=atom.geometry_out,
                                 scale=strand_scale.std_out, node_height=GRID,
                                 name="BackboneAtoms", label="BackboneAtoms")
        atom_material = SetMaterial(tree, location=at(4.8, -0.1),
                                    geometry=atoms.geometry_out,
                                    material=colors.out("Molecule"),
                                    node_height=GRID, name="PaintAtoms",
                                    label="PaintAtoms")

        join = JoinGeometry(tree, location=at(6.3, -1.1), node_height=GRID,
                            name="JoinBackbone", label="JoinBackbone")
        for piece in (pipe_material.geometry_out, atom_material.geometry_out):
            tree.links.new(piece, join.geometry_in)
        smooth = SetShadeSmooth(tree, location=at(7.8, -1.1),
                                geometry=join.geometry_out, node_height=GRID,
                                name="SmoothBackbone", label="SmoothBackbone")

        frame = Frame(tree, location=(27.1, 3.9), label="Strand Geometry",
                      node_height=GRID)
        frame.add([curve, strand_scale, colors, profile, pipe, pipe_material,
                   atom, atoms, atom_material, join, smooth])
        return smooth.geometry_out


class MovingTapeModifier(GeometryNodesModifier):
    """A tape of random bytes travelling through the shot, clipped at its edges.

    The port of ``video_bff/tmp.xml`` as the editor holds it now - the tree
    that took the place of the morphing one behind :class:`MorphModifier`. It
    draws the soup's tape the way the paper writes it down: a strip of square
    cells, one byte on each, sliding past the camera from left to right and
    vanishing at both sides of the frame.

    Four frames of nodes, the four the editor shows:

    ``ControlFrame``
        every constant: the two dimensions of the tape, the three numbers of
        the motion, the two edges it is cut off at, the offset that lifts a
        number clear of its cell, and the two materials.
    ``TapeCellInstantiation``
        one cell: a ``Grid`` of ``TapeSize`` square, turned into an instance so
        that it can be tilted as a whole, and dropped onto every point of the
        tape.
    ``NumberGeneration``
        the byte written on that cell: ``Value to String`` into ``String to
        Curves``, filled, moved onto the cell, extruded to give the digits some
        thickness and stood upright.
    ``TapeCutoff``
        the two comparisons that say whether a point has left the shot.

    **The tape.** A ``Mesh Line`` of ``TapeLength`` points from the origin to
    ``tape_span`` along x, so the cells sit ``tape_span / (TapeLength - 1)``
    apart - a hair over one unit for the defaults, which is what leaves a gap
    between cells 0.9 wide. The bytes are not read from anywhere: a ``Random
    Value`` in ``0…max_value`` draws one per point - its ``ID`` socket is left
    alone and falls back to the index - so every cell holds a different byte
    and holds the same one on every frame.

    **The motion.** Everything downstream of ``Scene Time`` is one straight
    line::

        x(t) = travel_distance * max(t - start_time, 0) / transition_time
               - travel_distance / 2

    - the tape starts half its travel to the left and moves right at
    ``travel_distance / transition_time``. Only the *lower* end is held: past
    ``start_time + transition_time`` the tape keeps going rather than stopping,
    which for the defaults it may, since it has left the frame long before
    (see :meth:`crossing_times`). The editor writes that out as six loose
    ``Math`` nodes and a ``Combine XYZ``; here it is one ``make_function``
    node, ``TapeShift``, holding the formula as it is written above. The whole
    animation is that one line, so a scene has nothing to keyframe - and
    nothing to keyframe is a problem of its own for ``render_with_skips``; see
    ``BffScene.moving_tape``.

    **The digits stand up.** ``Rotate Instances`` turns them by a quarter turn
    and nothing else, so they are upright however far the cells beneath them
    are laid back - ``cell_tilt`` tilts the tape alone. That is what lets the
    shot be taken head-on: the numbers face the camera squarely and the tape is
    the only thing in the frame that leans.

    **The clipping.** ``Delete Geometry`` throws away every *realized* point
    whose x lies outside ``[left_cutoff, right_cutoff]``. It runs after the for
    each zone rather than before, so it cuts the finished cells and digits and
    not the tape's points: a cell straddling the edge is chopped in half rather
    than kept or dropped whole. Put the two cutoffs just outside what the
    camera sees and the tape simply is not there beyond them - which is what
    keeps a hundred cells, most of them off screen, from being drawn.

    **The colour of a digit.** ``NumberMaterial`` is ``BaseMixing``, which
    ``appearance.textures.base_mixing`` builds: a voronoi texture quantized to
    the four colours the RNA bases are painted in. It varies along the
    ``CellSeed`` attribute the digits carry, and that is what the seed of
    ``NumberGeneration`` is for. Naming ``number_color`` a palette colour
    instead gives every digit the same one and leaves the attribute unread.

    :param tape_length: number of cells.
    :param tape_span: how far along x those cells are spread. Their spacing is
        this divided by ``tape_length - 1``.
    :param tape_size: width and height of one cell - the ``TapeSize`` value
        node of the editor's control frame.
    :param cell_tilt: angle the cells are laid back by, so that a camera
        looking along +y sees a face rather than an edge.
    :param number_size: height of the digits.
    :param number_offset: where the digits sit relative to their cell.
    :param number_depth: how far the digits are extruded.
    :param max_value: largest byte a cell can hold.
    :param seed: the ``Random Value`` seed - change it for a different tape.
    :param start_time: seconds before the tape starts moving.
    :param transition_time: seconds the tape takes for its whole travel. This
        is the dial that decides whether the bytes can be read on the way past.
    :param travel_distance: how far it travels, centred on the origin.
    :param left_cutoff: x below which nothing is drawn.
    :param right_cutoff: x above which nothing is drawn.
    :param tape_color: palette name for the cells.
    :param number_color: what the digits are drawn in - a palette name, or
        ``"BaseMixing"``, the material ``appearance.textures.base_mixing``
        builds out of a voronoi texture and the four colours the RNA bases are
        painted in. That is the default, and it is an argument rather than a
        decoration: a byte of the soup wearing the colours of the molecule says
        the two are the same stuff.
    :param number_seed: name of the attribute the digits carry their seed in -
        the cell's index plus the digit's place in the number, which is what
        gives the ``1`` and the ``7`` of a ``117`` different colours. ``None``
        drops the two nodes that work it out, for a material that has no use
        for it.
    """

    # Where the four frames of the editor sit. Everything inside one of them is
    # placed through _in_frame(<origin>), which is what turns the relative
    # coordinates an exported xml gives for a framed node back into the
    # absolute ones this file writes - see the note on _in_frame itself.
    CONTROL_FRAME = (-10.3, 5.0)
    CELL_FRAME = (-0.2, 5.4)
    NUMBER_FRAME = (-2.8, 3.6)
    CUTOFF_FRAME = (5.3, 1.8)

    def __init__(self, tape_length=50, tape_span=50.0, tape_size=0.9,
                 cell_tilt=0.31066858768463135, number_size=0.5,
                 number_offset=(0, 0, 0.3), number_depth=0.1,
                 max_value=255, seed=0,
                 start_time=1.0, transition_time=20.0, travel_distance=115.0,
                 left_cutoff=-14.0, right_cutoff=14.0,
                 tape_color="gray_1", number_color="BaseMixing",
                 number_seed="CellSeed",
                 name="MovingTape", **kwargs):
        self.tape_length = tape_length
        self.tape_span = tape_span
        self.tape_size = tape_size
        self.cell_tilt = cell_tilt
        self.number_size = number_size
        self.number_offset = Vector(number_offset)
        self.number_depth = number_depth
        self.max_value = max_value
        self.seed = seed
        self.start_time = start_time
        self.transition_time = transition_time
        self.travel_distance = travel_distance
        self.left_cutoff = left_cutoff
        self.right_cutoff = right_cutoff
        self.tape_color = tape_color
        self.number_color = number_color
        self.number_seed = number_seed
        self.kwargs = kwargs
        super().__init__(name=name, automatic_layout=False)

    # ----------------------------------------------------------------
    def crossing_times(self):
        """When the tape reaches the frame and when the last of it has left.

        The tape is a segment of length ``tape_span`` whose left end is at
        ``travel_distance * (t - start_time) / transition_time -
        travel_distance / 2``; it has something to show while its right end is
        past ``left_cutoff`` and its left end has not passed ``right_cutoff``.
        Both are worth knowing for a shot that should not open on an empty
        frame or hold on one - and they are not ``start_time`` and
        ``start_time + transition_time``: with the exported numbers the tape
        crosses in a little over half its travel and spends the rest of it off
        screen to the right.

        :return: ``(enter, leave)`` in seconds.
        """
        speed = self.travel_distance / self.transition_time
        start = -0.5 * self.travel_distance
        enter = self.start_time + (self.left_cutoff - self.tape_span - start) / speed
        leave = self.start_time + (self.right_cutoff - start) / speed
        # the tape may already reach into the frame at t = start_time, and it
        # does not move before then
        return max(enter, self.start_time), leave

    # ----------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        control = self._create_control_frame(tree)
        tape = self._create_tape(tree, control)

        # one turn per cell: Value to String needs a single value, and a string
        # is not a field - the same reason the machine's own cells are numbered
        # inside a zone (see BrainFuckSimpleModifier._create_cell_values)
        zone = ForEachZone(tree, location=(-4.0, 3.6), domain="POINT",
                           node_width=11.8, node_height=GRID, geometry=tape)
        zone.foreach_output.location = (7.8 * GRID, 3.8 * GRID)
        values = RandomValue(tree, location=(-5.0, 2.6), data_type="INT",
                             min=0, max=self.max_value, seed=self.seed,
                             node_height=GRID, name="CellValue")
        zone.add_socket(socket_type="INT", name="Value", value=values.std_out,
                        for_input=True)
        # the editor's zone carries two more input items that nothing inside it
        # reads; they are kept so that the tree round-trips
        zone.add_socket(socket_type="INT", name="Index")
        zone.add_socket(socket_type="VECTOR", name="Position")

        cells = self._create_cell_frame(tree, control, zone)
        painted = SetMaterial(tree, location=(4.0, 4.6), geometry=cells,
                              material=control["TapeMaterial"].std_out,
                              node_height=GRID, name="PaintTape")
        numbers = self._create_number_frame(tree, control, zone)

        joined = JoinGeometry(tree, location=(6.8, 3.8), node_height=GRID,
                              name="JoinCell")
        tree.links.new(painted.geometry_out, joined.geometry_in)
        tree.links.new(numbers, joined.geometry_in)
        tree.links.new(joined.geometry_out, zone.foreach_output.inputs["Geometry"])

        # realizing before the cut is what makes the cut a cut: the delete then
        # works on the points of the cells and digits themselves
        realize = RealizeInstances(tree, location=(9.2, 3.4),
                                   geometry=zone.geometry_out, node_height=GRID,
                                   name="RealizeTape")
        cut = DeleteGeometry(tree, location=(10.9, 2.9), domain="POINT", mode="ALL",
                             geometry=realize.geometry_out,
                             selection=self._create_cutoff_frame(tree, control),
                             node_height=GRID, name="CutOffTape")

        self.group_outputs.location = (11.9 * GRID, 2.8 * GRID)
        tree.links.new(cut.geometry_out, self.group_outputs.inputs["Geometry"])

    # ----------------------------------------------------------------
    def _create_control_frame(self, tree):
        """``ControlFrame``: every constant of the tape and of its motion.

        :return: ``{name: node}``, so that the frames downstream can pick the
            parameter they need by the name it carries in the editor.
        """
        at = _in_frame(self.CONTROL_FRAME)
        control = {
            "TapeSize": InputValue(tree, location=at(0.1, -0.1), value=self.tape_size,
                                   node_height=GRID, name="TapeSize"),
            "TapeLength": InputInteger(tree, location=at(0.1, -0.7),
                                       integer=self.tape_length,
                                       node_height=GRID, name="TapeLength"),
            "StartTime": InputValue(tree, location=at(0.1, -1.1), value=self.start_time,
                                    node_height=GRID, name="StartTime"),
            "TransitionTime": InputValue(tree, location=at(0.1, -1.6),
                                         value=self.transition_time,
                                         node_height=GRID, name="TransitionTime"),
            "TravelDistance": InputValue(tree, location=at(0.1, -2.2),
                                         value=self.travel_distance,
                                         node_height=GRID, name="TravelDistance"),
            "TapeMaterial": InputMaterial(tree, location=at(0.2, -3.5),
                                          material=self.tape_color,
                                          node_height=GRID, name="TapeMaterial",
                                          **self.kwargs),
            # the seed goes with the colour: a material that varies over the
            # geometry - BaseMixing - varies along this attribute, and one that
            # does not (a palette name) has no use for it and drops it
            "NumberMaterial": InputMaterial(tree, location=at(0.2, -3.9),
                                            material=self.number_color,
                                            node_height=GRID, name="NumberMaterial",
                                            **dict(self.kwargs,
                                                   attribute=self.number_seed)),
            "LeftCutoff": InputValue(tree, location=at(0.2, -4.5), value=self.left_cutoff,
                                     node_height=GRID, name="LeftCutoff"),
            "RightCutoff": InputValue(tree, location=at(0.2, -4.9), value=self.right_cutoff,
                                      node_height=GRID, name="RightCutoff"),
            "NumberOffset": InputVector(tree, location=at(0.1, -5.4),
                                        vector=self.number_offset,
                                        node_height=GRID, name="NumberOffset"),
        }
        for source in ("TapeMaterial", "NumberMaterial"):
            self.materials.append(control[source].node.material)

        frame = Frame(tree, location=self.CONTROL_FRAME, label="ControlFrame",
                      node_height=GRID)
        frame.add(list(control.values()))
        return control

    # ----------------------------------------------------------------
    def _create_tape(self, tree, control):
        """The row of points the cells are dropped onto, where it is right now.

        :return: the geometry socket of the moved tape.
        """
        line = MeshLine(tree, location=(-5.8, 3.9), mode="END_POINTS", hide=True,
                        count=control["TapeLength"].std_out,
                        start_location=Vector([0, 0, 0]),
                        end_location=Vector([self.tape_span, 0, 0]),
                        node_height=GRID, name="Tape")
        move = SetPosition(tree, location=(-4.8, 4.0), geometry=line.geometry_out,
                           offset=self._create_motion(tree, control),
                           node_height=GRID, name="MoveTape")
        return move.geometry_out

    # ----------------------------------------------------------------
    def _create_motion(self, tree, control):
        """How far the tape has travelled by now, as a vector along x.

        The editor spells this out as six loose nodes between the control
        frame and the tape - a subtract, a maximum, a divide, two multiplies
        and an add, into a ``Combine XYZ``. It is one formula and it is
        written as one here, which also settles the trap in the middle of that
        chain: ``Maximum`` has to be given a second operand of *zero*, and zero
        is exactly the value the wrappers read as "not given" and leave at
        blender's default of 0.5.

        :return: the vector socket to offset the tape by.
        """
        clock = SceneTime(tree, location=(-8.8, 3.2), std_out="Seconds",
                          node_height=GRID, name="Clock")
        shift = make_function(tree, name="TapeShift",
                              functions={
                                  "shift": ["time,start,-,0,max,transition,/,"
                                            "distance,*,distance,2,/,-", "0", "0"],
                              },
                              inputs=["time", "start", "transition", "distance"],
                              outputs=["shift"],
                              scalars=["time", "start", "transition", "distance"],
                              vectors=["shift"], hide=True)
        # make_function scales y by 100 rather than by the 200 everything else
        # in this class uses, so its own location argument cannot be used
        shift.location = (-6.4 * GRID, 3.2 * GRID)
        tree.links.new(clock.std_out, shift.inputs["time"])
        tree.links.new(control["StartTime"].std_out, shift.inputs["start"])
        tree.links.new(control["TransitionTime"].std_out, shift.inputs["transition"])
        tree.links.new(control["TravelDistance"].std_out, shift.inputs["distance"])
        return shift.outputs["shift"]

    # ----------------------------------------------------------------
    def _create_cell_frame(self, tree, control, zone):
        """``TapeCellInstantiation``: one square cell on every point of the tape.

        The cell is turned into an instance *before* it is tilted, so that
        ``Rotate Instances`` turns the whole square about the tape rather than
        each of its points about itself - the grid has a hundred of them and a
        field would tilt them one by one into nothing.

        :return: the geometry socket of the cells.
        """
        at = _in_frame(self.CELL_FRAME)
        # the reroute is the editor's, and it sits outside the frame: one value
        # feeding both sides of the square
        route = Reroute(tree, location=(-1.0, 4.2), ins=control["TapeSize"].std_out,
                        node_height=GRID, name="TapeSizeRoute")
        cell = Grid(tree, location=at(0.1, -0.3), size_x=route.std_out,
                    size_y=route.std_out, vertices_x=10, vertices_y=10,
                    node_height=GRID, name="TapeCell")
        instance = GeometryToInstance(tree, location=at(0.9, -0.2), hide=True,
                                      node_height=GRID, name="CellInstance")
        tree.links.new(cell.geometry_out, instance.geometry_in)
        tilt = RotateInstances(tree, location=at(1.9, -0.3), hide=True,
                               instances=instance.geometry_out,
                               rotation=[self.cell_tilt, 0, 0], local_space=False,
                               node_height=GRID, name="TiltCell")
        cells = InstanceOnPoints(tree, location=at(3.1, -0.1), points=zone.element,
                                 instance=tilt.geometry_out, node_height=GRID,
                                 name="CellsOnTape")

        frame = Frame(tree, location=self.CELL_FRAME,
                      label="TapeCellInstantiation", node_height=GRID)
        frame.add([cell, instance, tilt, cells])
        return cells.geometry_out

    # ----------------------------------------------------------------
    def _create_number_frame(self, tree, control, zone):
        """``NumberGeneration``: the byte of this cell, written on it.

        ``Sample Index`` is what puts the digits on the cell: inside the outer
        zone ``Element`` is the single point being worked on, so reading its
        ``Position`` at index 0 - plus ``NumberOffset``, which lifts the digits
        off the face - is where this cell is. The digits are built at the
        origin and moved there, rather than being instanced onto the point,
        because they have to be filled and extruded first.

        **A zone of its own for the glyphs.** ``String to Curves`` hands back
        one *instance* per character, and everything from ``Fill Curve`` on is
        wrapped in a second for each zone that walks them one at a time, on the
        ``INSTANCE`` domain. That is what lets the ``1`` and the ``7`` of a
        ``117`` be told apart: the seed the ``BaseMixing`` material picks its
        colour from is the cell's index *plus the glyph's*, so the digits of one
        number are consecutive seeds rather than one seed shared, and each
        digit comes out in a base colour of its own.

        What the seed must not be is a position. The tape travels, so a colour
        that depends on where a digit *is* changes about ten times a second on
        the way past; an index changes never.

        :return: the geometry socket of the digits.
        """
        at = _in_frame(self.NUMBER_FRAME)
        digits = ValueToString(tree, location=at(0.2, -1.1), data_type="INT",
                               value=zone.foreach_input.outputs["Value"],
                               node_height=GRID, name="CellDigits")
        curves = StringToCurves(tree, location=at(1.0, -0.6),
                                string=digits.std_out, size=self.number_size,
                                align_x="CENTER", align_y="MIDDLE",
                                pivot_mode="MIDPOINT", node_height=GRID,
                                name="NumberCurves")

        # one turn per character of the number
        glyphs = ForEachZone(tree, location=at(2.2, -0.3), domain="INSTANCE",
                             node_width=6.6, node_height=GRID,
                             geometry=curves.geometry_out)
        glyphs.foreach_output.location = (at(8.8, -0.2)[0] * GRID,
                                          at(8.8, -0.2)[1] * GRID)

        fill = FillCurve(tree, location=at(3.2, -0.8), mode="Triangles",
                         curve=glyphs.element, node_height=GRID,
                         name="FillNumber")

        where = Position(tree, location=at(0.1, -2.0), node_height=GRID,
                         name="CellPosition")
        raised = VectorMath(tree, location=at(1.2, -1.9), operation="ADD",
                            inputs0=where.std_out,
                            inputs1=control["NumberOffset"].std_out,
                            node_height=GRID, name="NumberPlacement")
        sample = SampleIndex(tree, location=at(2.1, -1.6), hide=True,
                             data_type="FLOAT_VECTOR", domain="POINT",
                             geometry=zone.element, value=raised.std_out,
                             node_height=GRID, name="SampleCellPosition")
        placed = SetPosition(tree, location=at(4.2, -0.5), geometry=fill.geometry_out,
                             offset=sample.std_out, node_height=GRID,
                             name="PlaceNumber")
        thick = ExtrudeMesh(tree, location=at(5.2, -0.4), mode="FACES",
                            mesh=placed.geometry_out, offset_scale=self.number_depth,
                            node_height=GRID, name="ThickenNumber")
        # a quarter turn about x, and only that: the digits stand upright out
        # of the tape's plane whatever the cells under them are tilted by
        upright = RotateInstances(tree, location=at(6.1, -0.5),
                                  instances=thick.geometry_out,
                                  rotation=[pi / 2, 0, 0], node_height=GRID,
                                  name="StandNumberUp")
        painted = SetMaterial(tree, location=at(7.0, -0.5),
                              geometry=upright.geometry_out,
                              material=control["NumberMaterial"].std_out,
                              node_height=GRID, name="PaintNumber")

        last = painted
        pieces = [digits, curves, glyphs, fill, where, raised, sample, placed,
                  thick, upright, painted]
        if self.number_seed:
            # which colour of the four this digit wears: the cell it belongs to,
            # offset by its place in the number
            seed = MathNode(tree, location=at(6.9, -1.3), operation="ADD",
                            inputs0=zone.index, inputs1=glyphs.index,
                            node_height=GRID, name="SeedIndex", label="")
            last = StoredNamedAttribute(tree, location=at(7.9, -0.3),
                                        data_type="FLOAT", domain="POINT",
                                        name=self.number_seed, value=seed.std_out,
                                        node_height=GRID, label="SeedNumber")
            tree.links.new(painted.geometry_out, last.geometry_in)
            pieces += [seed, last]
        tree.links.new(last.geometry_out, glyphs.foreach_output.inputs["Geometry"])

        frame = Frame(tree, location=self.NUMBER_FRAME, label="NumberGeneration",
                      node_height=GRID, mute=True)
        frame.add(pieces)
        return glyphs.geometry_out

    # ----------------------------------------------------------------
    def _create_cutoff_frame(self, tree, control):
        """``TapeCutoff``: is this point outside the shot?

        :return: the boolean socket to delete by.
        """
        at = _in_frame(self.CUTOFF_FRAME)
        where = Position(tree, location=at(0.1, -0.7), node_height=GRID,
                         name="TapePosition")
        along = SeparateXYZ(tree, location=at(1.0, -0.4), vector=where.std_out,
                            node_height=GRID, name="TapeX")
        left = CompareNode(tree, location=at(1.9, -0.1), operation="LESS_THAN",
                           data_type="FLOAT", inputs0=along.x,
                           inputs1=control["LeftCutoff"].std_out,
                           node_height=GRID, name="BeyondLeft")
        right = CompareNode(tree, location=at(2.0, -1.0), operation="GREATER_THAN",
                            data_type="FLOAT", inputs0=along.x,
                            inputs1=control["RightCutoff"].std_out,
                            node_height=GRID, name="BeyondRight")
        gone = BooleanMath(tree, location=at(2.9, -0.5), operation="OR",
                           inputs0=left.std_out, inputs1=right.std_out,
                           node_height=GRID, name="OffScreen")

        frame = Frame(tree, location=self.CUTOFF_FRAME, label="TapeCutoff",
                      node_height=GRID)
        frame.add([where, along, left, right, gone])
        return gone.std_out


class LifeOnEarthModifier(GeometryNodesModifier):
    """"Life on Earth", written flat and then wrapped into a graticule.

    The port of ``video_bff/tmp.xml`` as the editor holds it now - the tree
    that took the place of the moving tape. Two words are set as curves, the
    letters grow in one at a time, and then every glyph outline inflates into
    a circle on a sphere: the letters of ``Life on`` become the circles of
    longitude, the letters of ``Earth`` the circles of latitude, and what is
    left standing when the move is over is a wireframe globe made of nothing
    but the sentence it started as. A solid ball arrives underneath it in the
    last fraction of a second, so the graticule finishes on a planet rather
    than on a cage.

    **A letter is a spline, not a character.** ``LattitudeIndex`` and
    ``LongitudeIndex`` are stored on the ``CURVE`` domain after the glyph
    instances are realized, so what they number is *outlines*: the ``e`` of
    ``Life`` is two of them (its body and its counter) and contributes two
    meridians. That is the editor's own choice and it is kept - it is also
    what makes the graticule dense enough to read as one, since ``Life on``
    only has six characters.

    **How a glyph becomes a circle.** Every outline is resampled to
    ``Resolution`` points, so point ``i`` of a spline is its ``i mod
    Resolution``-th sample whichever spline it is on. That is the angle round
    the circle; which circle comes from the letter's index::

        meridian:  phi = (LattitudeIndex - 1) * pi / letters("Life on")
                   theta = (i mod Resolution) * 2 pi / Resolution
        parallel:  theta = LongitudeIndex * pi / (letters("Earth") + 1)
                   phi = (i mod Resolution) * 2 pi / Resolution

        p = Radius * (cos phi sin theta, cos theta, sin phi sin theta)

    with the poles on **y**, so an object carrying this modifier has to be
    turned a quarter turn about x for them to stand upright - see
    ``BffScene.how_on_earth``. A point is then linearly blended from where its
    letter drew it to where its circle wants it,

        ``position = (1 - lambda) * flat + lambda * p``,

    over ``TransformDuration`` from ``TransformTime``. Because both ends are
    written into one ``Set Position`` there is nothing to keyframe - the whole
    animation lives in ``Scene Time`` inside the graph, which is a problem of
    its own for ``render_with_skips``; see :meth:`BffScene.how_on_earth`.

    **The letters grow rather than pop.** The editor's tree simply deleted
    every letter whose moment had not come, so the sentence arrived one glyph
    at a time in full size. Here each letter is given the whole slot it used
    to wait through: letter ``i`` of ``Life on`` grows from nothing to full
    size over ``[StartTime + (i-1) dt, StartTime + i dt]`` with ``dt =
    TransitionTime / letters``, so it is complete at exactly the moment it
    used to appear at and the word is still finished at ``StartTime +
    TransitionTime``. ``Earth`` follows on the same terms one
    ``TransitionTime`` later, over ``TransitionTime2``.

    Growing means scaling about the glyph's own centre, and that centre is
    not something the realized curve knows any more - by then a letter is
    just a run of splines among all the others. So it is written down while
    the glyphs are still *instances*, one ``Store Named Attribute`` on the
    ``INSTANCE`` domain (:attr:`PIVOT`) recording each glyph's placement,
    which ``Realize Instances`` then copies onto every point of that glyph.
    ``Earth``'s pivots carry ``word_offset`` themselves, because the
    ``Transform Geometry`` that shifts the word runs after the store and does
    not touch a named attribute.

    Eleven frames of nodes, the editor's eleven - and they are built in this
    order, because blender names a frame by when it was made and the xml
    refers to them by that name:

    ``ControlPanel``
        every constant: the four times, the resolution, the radius of the
        globe and of the tubes, and the two materials.
    ``Create Geometry``
        the two words as curves, realized and numbered. The ``Resample
        Curve`` that gives every outline the same point count sits just
        outside it, on the way to the reroute the rest of the graph reads.
    ``Lambda``
        the one ramp the whole transformation runs on, and the clock the
        letters are timed against.
    ``Grow Life On`` / ``Earth``
        each word's own schedule, and the two ``Delete Geometry`` nodes that
        leave only that word's letters, only once they have started.
    ``Circles Of Longitude`` / ``Circles Of Latitude``
        where a point goes on the sphere.
    ``Original Position`` (twice)
        where it came from, scaled about its glyph's pivot by how far that
        letter has grown.
    ``Globe``
        the ball, swelling to ``Radius - TubeRadius`` over the last
        ``GlobeLead`` seconds of the transformation, so that it arrives just
        as the graticule settles and sits exactly inside the tubes.
    ``CurveGeometry``
        both words joined and swept into tube.

    :param first_word: the word that becomes the meridians.
    :param second_word: the word that becomes the parallels.
    :param word_offset: where ``second_word`` is set relative to
        ``first_word`` - the two are separate ``String to Curves`` nodes on
        one baseline, so this is what spaces them into one sentence.
    :param text_size: cap height of the lettering.
    :param font: name of a loaded Blender font.
    :param start_time: when the first letter starts growing.
    :param transition_time: seconds ``first_word`` takes to write itself on.
    :param transition_time2: the same for ``second_word``, which follows it.
    :param transform_time: when the letters start leaving the page.
    :param transform_duration: seconds they take to reach the sphere.
    :param resolution: points every glyph outline is resampled to, and so the
        number of segments each circle of the graticule is drawn with.
    :param radius: radius of the globe the letters end up on.
    :param tube_radius: radius of the tube every outline is swept into.
    :param profile_resolution: segments of that tube.
    :param globe_lead: how long before the end of the transformation the
        solid ball starts to appear, in seconds.
    :param globe_segments: meridional resolution of the ball.
    :param globe_rings: latitudinal resolution of the ball.
    :param letter_color: palette name for the lettering.
    :param letter_emission: emission strength of the lettering material.
        These scenes sit on a black background and are lit mostly by their
        own emission, which is why this defaults to full rather than to the
        ``**kwargs`` the ball takes.
    :param globe_color: palette name for the ball.
    """

    # Where the eleven frames of the editor sit. Everything inside one of them
    # is placed through _in_frame(<origin>), which turns the relative
    # coordinates an exported xml gives for a framed node back into the
    # absolute ones this file writes - see the note on _in_frame itself.
    # They are also built in this order, because blender names a frame by when
    # it was made and the xml refers to them by that name.
    CONTROL_FRAME = (0.1, 2.5)
    CREATE_FRAME = (-7.5, 1.9)
    LAMBDA_FRAME = (2.8, 1.6)
    GROW_FRAME = (3.0, 3.7)
    EARTH_FRAME = (3.2, -1.1)
    MERIDIAN_FRAME = (9.7, 2.2)
    PARALLEL_FRAME = (9.5, -0.4)
    FLAT_LIFE_FRAME = (9.9, 4.1)
    FLAT_EARTH_FRAME = (10.0, -2.4)
    GLOBE_FRAME = (14.3, 1.7)
    TUBE_FRAME = (15.2, -0.3)

    #: the glyph's own centre, stored while the glyphs are still instances so
    #: that a letter can be grown about it once they are not
    PIVOT = "GlyphPivot"
    #: the two per-outline counters, spelled the way the editor spells them -
    #: they are the keys the tree reads itself by, so the typo stays
    LATITUDE = "LattitudeIndex"
    LONGITUDE = "LongitudeIndex"

    def __init__(self, first_word="Life on", second_word="Earth",
                 word_offset=(5.8, 0, 0), text_size=2.0, font="Bfont Regular",
                 start_time=1.0, transition_time=2.0, transition_time2=1.5,
                 transform_time=6.0, transform_duration=3.0,
                 resolution=100, radius=10.0,
                 tube_radius=0.03, profile_resolution=8,
                 globe_lead=0.1, globe_segments=64, globe_rings=32,
                 letter_color="example", letter_emission=1.0,
                 globe_color="joker", name="LifeOnEarth", **kwargs):
        self.first_word = first_word
        self.second_word = second_word
        self.word_offset = Vector(word_offset)
        self.text_size = text_size
        self.font = font
        self.start_time = start_time
        self.transition_time = transition_time
        self.transition_time2 = transition_time2
        self.transform_time = transform_time
        self.transform_duration = transform_duration
        self.resolution = resolution
        self.radius = radius
        self.tube_radius = tube_radius
        self.profile_resolution = profile_resolution
        self.globe_lead = globe_lead
        self.globe_segments = globe_segments
        self.globe_rings = globe_rings
        self.letter_color = letter_color
        self.letter_emission = letter_emission
        self.globe_color = globe_color
        self.kwargs = kwargs
        super().__init__(name=name, automatic_layout=False)

    # ----------------------------------------------------------------
    @property
    def written(self):
        """When the last letter of ``second_word`` has finished growing."""
        return self.start_time + self.transition_time + self.transition_time2

    @property
    def transform_end(self):
        """When the graticule has settled and the ball is fully there."""
        return self.transform_time + self.transform_duration

    def timeline(self):
        """The four moments a shot has to be cut around.

        :return: ``(written, transform_time, globe_in, transform_end)`` in
            seconds - the sentence complete, the letters starting to leave it,
            the ball starting to appear, and everything in place.
        """
        return (self.written, self.transform_time,
                self.transform_end - self.globe_lead, self.transform_end)

    # ----------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        control = self._create_control_panel_frame(tree)
        letters, counts = self._create_geometry_frame(tree, control)
        ramp, clock = self._create_lambda_frame(tree, control)

        # one ramp, two halves of the graph a long way apart - the editor's
        # own pair of reroutes, and the reason the two Original Position
        # frames each read their own socket rather than sharing one noodle
        to_life = Reroute(tree, location=(9.5, 2.4), ins=ramp, node_height=GRID,
                          name="LambdaToLife")
        to_earth = Reroute(tree, location=(9.5, -2.2), ins=ramp, node_height=GRID,
                           name="LambdaToEarth")

        life_geo, life_grow = self._create_grow_life_on_frame(
            tree, control, letters, counts["life"], clock)
        earth_geo, earth_grow = self._create_earth_frame(
            tree, control, letters, counts["earth"], clock)

        meridian = self._create_circles_of_longitude_frame(
            tree, control, counts["life"], to_life.std_out)
        parallel = self._create_circles_of_latitude_frame(
            tree, control, counts["earth"], to_earth.std_out)

        flat_life = self._create_original_position_frame(
            tree, self.FLAT_LIFE_FRAME, life_grow, to_life.std_out, "Life")
        flat_earth = self._create_original_position_frame(
            tree, self.FLAT_EARTH_FRAME, earth_grow, to_earth.std_out, "Earth")

        # (1 - lambda) * where the letter drew it + lambda * where its circle
        # wants it: one Set Position holds the whole transformation
        life_at = VectorMath(tree, location=(12.2, 1.4), operation="ADD",
                             inputs0=flat_life, inputs1=meridian,
                             node_height=GRID, name="LifePosition")
        life_moved = SetPosition(tree, location=(13.3, 1.6), geometry=life_geo,
                                 position=life_at.std_out, node_height=GRID,
                                 name="MoveLifeOntoGlobe")
        earth_at = VectorMath(tree, location=(12.5, -0.4), operation="ADD",
                              inputs0=flat_earth, inputs1=parallel,
                              node_height=GRID, name="EarthPosition")
        earth_moved = SetPosition(tree, location=(13.4, -0.1), geometry=earth_geo,
                                  position=earth_at.std_out, node_height=GRID,
                                  name="MoveEarthOntoGlobe")

        globe = self._create_globe_frame(tree, control)
        tubes = self._create_curve_geometry_frame(
            tree, control, life_moved.geometry_out, earth_moved.geometry_out)

        joined = JoinGeometry(tree, location=(19.6, -0.7), node_height=GRID,
                              name="JoinGlobe")
        tree.links.new(tubes, joined.geometry_in)
        tree.links.new(globe, joined.geometry_in)

        self.group_outputs.location = (20.6 * GRID, -0.7 * GRID)
        tree.links.new(joined.geometry_out, self.group_outputs.inputs["Geometry"])

    # ----------------------------------------------------------------
    def _create_control_panel_frame(self, tree):
        """``ControlPanel``: every constant of the sentence and of the globe.

        :return: ``{name: node}``, so that the frames downstream can pick the
            parameter they need by the name it carries in the editor.
        """
        at = _in_frame(self.CONTROL_FRAME)
        control = {
            "StartTime": InputValue(tree, location=at(0.1, -0.1),
                                    value=self.start_time, node_height=GRID,
                                    name="StartTime"),
            "TransitionTime": InputValue(tree, location=at(0.2, -0.6),
                                         value=self.transition_time,
                                         node_height=GRID, name="TransitionTime"),
            "TransitionTime2": InputValue(tree, location=at(0.2, -0.9),
                                          value=self.transition_time2,
                                          node_height=GRID, name="TransitionTime2"),
            "Resolution": InputInteger(tree, location=at(0.2, -1.4),
                                       integer=self.resolution,
                                       node_height=GRID, name="Resolution"),
            "TransformTime": InputValue(tree, location=at(0.2, -1.8),
                                        value=self.transform_time,
                                        node_height=GRID, name="TransformTime"),
            "TransformDuration": InputValue(tree, location=at(0.2, -2.1),
                                            value=self.transform_duration,
                                            node_height=GRID,
                                            name="TransformDuration"),
            "Radius": InputValue(tree, location=at(0.1, -2.7), value=self.radius,
                                 node_height=GRID, name="Radius"),
            "TubeRadius": InputValue(tree, location=at(0.1, -3.1),
                                     value=self.tube_radius, node_height=GRID,
                                     name="TubeRadius"),
            "GlobeLead": InputValue(tree, location=at(0.1, -3.5),
                                    value=self.globe_lead, node_height=GRID,
                                    name="GlobeLead"),
        }

        # **self.kwargs carries things like `emission=0.6` through to every
        # material - the same forwarding the other modifiers of this file do.
        # The lettering overrides it: it is the thing the shot is about and it
        # is read on black, so it is emissive whatever the ball is.
        control["LetterMaterial"] = InputMaterial(
            tree, location=at(0.1, -3.9), material=self.letter_color,
            node_height=GRID, name="LetterMaterial",
            **dict(self.kwargs, emission=self.letter_emission))
        control["GlobeMaterial"] = InputMaterial(
            tree, location=at(0.1, -4.3), material=get_texture(self.globe_color, **self.kwargs),
            node_height=GRID, name="GlobeMaterial", **self.kwargs)
        for key in ("LetterMaterial", "GlobeMaterial"):
            self.materials.append(control[key].node.material)

        frame = Frame(tree, location=self.CONTROL_FRAME, label="ControlPanel",
                      node_height=GRID)
        frame.add(list(control.values()))
        return control

    # ----------------------------------------------------------------
    def _create_geometry_frame(self, tree, control):
        """``Create Geometry``: the sentence, numbered outline by outline.

        The two words are set separately so that they can be timed
        separately, and joined only once each has been stamped with a counter
        of its own - ``LattitudeIndex`` on ``Life on``, ``LongitudeIndex`` on
        ``Earth``. After the join, a curve carrying a zero in one of them is
        exactly a curve belonging to the other word, which is how the two
        halves of the graph tell their own letters apart (see the pair of
        ``Delete Geometry`` nodes in each).

        The counters are ``Index + 1`` rather than ``Index``, so that zero can
        mean "not this word" - and so that the ``Attribute Statistic`` reading
        their maximum comes back with the *number* of outlines rather than
        with the last one's index.

        :return: ``(curves, counts)`` - the resampled sentence, and the two
            outline counts keyed ``"life"`` and ``"earth"``.
        """
        at = _in_frame(self.CREATE_FRAME)
        first = InputString(tree, location=at(0.9, -2.2), string=self.first_word,
                            node_height=GRID, name="FirstWord")
        second = InputString(tree, location=at(0.1, -2.9), string=self.second_word,
                             node_height=GRID, name="SecondWord")
        life = StringToCurves(tree, location=at(1.8, -1.5), string=first.std_out,
                              size=self.text_size, font=self.font, align_x="LEFT",
                              align_y="TOP_BASELINE", pivot_mode="MIDPOINT",
                              node_height=GRID, name="LifeCurves")
        earth = StringToCurves(tree, location=at(1.0, -2.7), string=second.std_out,
                               size=self.text_size, font=self.font, align_x="LEFT",
                               align_y="TOP_BASELINE", pivot_mode="MIDPOINT",
                               node_height=GRID, name="EarthCurves")

        # A glyph's own centre, written down while it is still an instance -
        # nothing downstream could work it out again, since by then a letter
        # is just a run of splines among all the others. It takes both of
        # String to Curves' answers: `Position` on the INSTANCE domain is
        # where the glyph sits on the baseline, and the node's `Pivot Point`
        # output is the middle of the glyph *within* that instance (which is
        # what `pivot_mode="MIDPOINT"` is there to say). Their sum is the
        # letter's middle in the sentence's own frame, and it is what a letter
        # grows about. Realize Instances then copies the attribute onto every
        # point of the geometry it realizes, so each point comes out knowing
        # which letter it belongs to and where that letter's middle is.
        life_origin = Position(tree, location=at(1.3, -0.2), node_height=GRID,
                               hide=True, name="LifeGlyphOrigin")
        life_centre = VectorMath(tree, location=at(2.1, -0.2), operation="ADD",
                                 inputs0=life_origin.std_out,
                                 inputs1=life.pivot_point, node_height=GRID,
                                 hide=True, name="LifeGlyphCentre")
        life_pivot = StoredNamedAttribute(tree, location=at(2.9, -0.2),
                                          data_type="FLOAT_VECTOR", domain="INSTANCE",
                                          name=self.PIVOT, value=life_centre.std_out,
                                          node_height=GRID, hide=True,
                                          label="StoreLifePivot")
        tree.links.new(life.geometry_out, life_pivot.geometry_in)
        realize_life = RealizeInstances(tree, location=at(2.9, -1.3),
                                        geometry=life_pivot.geometry_out,
                                        node_height=GRID, name="RealizeLife")

        life_index = Index(tree, location=at(3.8, -1.5), node_height=GRID,
                           name="LifeOutline")
        life_number = MathNode(tree, location=at(4.7, -1.8), operation="ADD",
                               inputs0=life_index.std_out, inputs1=1.0,
                               node_height=GRID, name="LifeOutlineNumber")
        life_stored = StoredNamedAttribute(tree, location=at(5.5, -1.0),
                                           data_type="INT", domain="CURVE",
                                           name=self.LATITUDE,
                                           value=life_number.std_out,
                                           node_height=GRID,
                                           label="StoreLattitudeIndex")
        tree.links.new(realize_life.geometry_out, life_stored.geometry_in)

        # `Earth`'s pivots have to carry the offset themselves: the Transform
        # Geometry that shifts the word runs after the store and moves
        # positions, not named attributes
        earth_origin = Position(tree, location=at(0.1, -5.1), node_height=GRID,
                                hide=True, name="EarthGlyphOrigin")
        earth_centre = VectorMath(tree, location=at(0.9, -5.1), operation="ADD",
                                  inputs0=earth_origin.std_out,
                                  inputs1=earth.pivot_point, node_height=GRID,
                                  hide=True, name="EarthGlyphCentre")
        earth_shifted_origin = VectorMath(tree, location=at(1.7, -5.1),
                                          operation="ADD",
                                          inputs0=earth_centre.std_out,
                                          inputs1=self.word_offset,
                                          node_height=GRID, hide=True,
                                          name="EarthGlyphPlacement")
        earth_pivot = StoredNamedAttribute(tree, location=at(2.5, -5.1),
                                           data_type="FLOAT_VECTOR",
                                           domain="INSTANCE", name=self.PIVOT,
                                           value=earth_shifted_origin.std_out,
                                           node_height=GRID, hide=True,
                                           label="StoreEarthPivot")
        tree.links.new(earth.geometry_out, earth_pivot.geometry_in)
        realize_earth = RealizeInstances(tree, location=at(2.1, -2.8),
                                         geometry=earth_pivot.geometry_out,
                                         node_height=GRID, name="RealizeEarth")
        placed = TransformGeometry(tree, location=at(2.9, -2.8),
                                   geometry=realize_earth.geometry_out,
                                   translation=self.word_offset,
                                   node_height=GRID, name="PlaceEarth")

        earth_index = Index(tree, location=at(3.7, -3.8), node_height=GRID,
                            name="EarthOutline")
        earth_number = MathNode(tree, location=at(4.6, -3.4), operation="ADD",
                                inputs0=earth_index.std_out, inputs1=1.0,
                                node_height=GRID, name="EarthOutlineNumber")
        earth_stored = StoredNamedAttribute(tree, location=at(5.5, -2.6),
                                            data_type="INT", domain="CURVE",
                                            name=self.LONGITUDE,
                                            value=earth_number.std_out,
                                            node_height=GRID,
                                            label="StoreLongitudeIndex")
        tree.links.new(placed.geometry_out, earth_stored.geometry_in)

        life_count = AttributeStatistic(tree, location=at(6.7, -0.1),
                                        data_type="FLOAT", domain="CURVE",
                                        geometry=life_stored.geometry_out,
                                        attribute=life_number.std_out,
                                        std_out="Max", node_height=GRID,
                                        name="LifeOutlineCount")
        earth_count = AttributeStatistic(tree, location=at(6.8, -2.6),
                                         data_type="FLOAT", domain="CURVE",
                                         geometry=earth_stored.geometry_out,
                                         attribute=earth_number.std_out,
                                         std_out="Max", node_height=GRID,
                                         name="EarthOutlineCount")

        joined = JoinGeometry(tree, location=at(6.7, -2.0), node_height=GRID,
                              name="JoinWords")
        tree.links.new(earth_stored.geometry_out, joined.geometry_in)
        tree.links.new(life_stored.geometry_out, joined.geometry_in)
        frame = Frame(tree, location=self.CREATE_FRAME, label="Create Geometry",
                      node_height=GRID)
        frame.add([first, second, life, earth, life_origin, life_centre,
                   life_pivot, realize_life, life_index, life_number,
                   life_stored, earth_origin, earth_centre,
                   earth_shifted_origin, earth_pivot, realize_earth, placed,
                   earth_index, earth_number, earth_stored, life_count,
                   earth_count, joined])

        # every outline to the same number of points, which is what lets
        # `Index mod Resolution` be "how far round this circle am I". It sits
        # outside the frame, on the way out of it: what the rest of the graph
        # reads is the *resampled* sentence, and the two nodes that say so
        # belong with the reroute rather than with the lettering
        sampled = ResampleCurve(tree, location=(2.0, 0.2), mode="Count",
                                curve=joined.geometry_out,
                                count=control["Resolution"].std_out,
                                node_height=GRID, name="ResampleOutlines")
        # the sentence leaves over a reroute, because both halves of the graph
        # start from it
        route = Reroute(tree, location=(2.9, 0.0), ins=sampled.geometry_out,
                        node_height=GRID, name="Sentence")
        return route.std_out, {"life": life_count.std_out,
                               "earth": earth_count.std_out}

    # ----------------------------------------------------------------
    def _create_lambda_frame(self, tree, control):
        """``Lambda``: the ramp the whole transformation runs on.

        ``lambda = min(max(t - TransformTime, 0), TransformDuration) /
        TransformDuration`` - zero until the move starts, one once it is over,
        and the straight line in between. The editor spells it out as five
        loose ``Math`` nodes; it is one formula and it is written as one here.

        The clock the *letters* are timed against lives here too, next to the
        one the ramp reads. They are two ``Scene Time`` nodes rather than one
        because each is wired to only its own half of the graph, but keeping
        them together is what makes this frame "everything that reads the
        clock" rather than just the ramp.

        :return: ``(lambda, seconds)`` - the ramp in ``0..1``, and the letter
            clock the two word frames time themselves against.
        """
        at = _in_frame(self.LAMBDA_FRAME)
        letters = SceneTime(tree, location=at(0.1, -0.1), std_out="Seconds",
                            node_height=GRID, name="LetterClock")
        clock = SceneTime(tree, location=at(0.1, -0.6), std_out="Seconds",
                          node_height=GRID, name="TransformClock")
        ramp = make_function(
            tree, name="Lambda",
            functions={"lam": "time,transformTime,-,0,max,duration,min,duration,/"},
            inputs=["time", "transformTime", "duration"], outputs=["lam"],
            scalars=["time", "transformTime", "duration", "lam"], hide=True)
        # make_function scales y by 100 rather than by the 200 everything else
        # in this class uses, so its own location argument cannot be used
        ramp.location = tuple(coordinate * GRID for coordinate in at(1.6, -0.3))
        tree.links.new(clock.std_out, ramp.inputs["time"])
        tree.links.new(control["TransformTime"].std_out, ramp.inputs["transformTime"])
        tree.links.new(control["TransformDuration"].std_out, ramp.inputs["duration"])

        frame = Frame(tree, location=self.LAMBDA_FRAME, label="Lambda",
                      node_height=GRID)
        frame.add([letters, clock, ramp])
        return ramp.outputs["lam"], letters.std_out

    # ----------------------------------------------------------------
    def _create_letter_clock(self, tree, control, count, location, name, offset):
        """One word's writing-on schedule, as a formula.

        Letter ``i`` (counting from one, which is what the stored index is)
        owns the slot ``[offset + (i-1) dt, offset + i dt]``, where ``dt`` is
        the word's transition time divided by how many outlines it has. It
        grows across that slot and is complete at the end of it - which is the
        moment the editor's tree made it appear at, so the word is still
        finished exactly when it always was.

        :param offset: the socket the word's own clock starts from -
            ``StartTime`` for the first word, ``StartTime + TransitionTime``
            for the second.
        :return: the group node, with a ``begin`` and a ``grow`` output.
        """
        schedule = make_function(
            tree, name=name,
            aux_functions={"dt": "transition,count,/"},
            functions={
                "begin": "offset,index,1,-,dt,*,+",
                "grow": "time,offset,index,1,-,dt,*,+,-,dt,/,0,max,1,min",
            },
            inputs=["time", "offset", "transition", "count", "index"],
            outputs=["begin", "grow"],
            scalars=["time", "offset", "transition", "count", "index", "dt",
                     "begin", "grow"], hide=True)
        schedule.location = tuple(coordinate * GRID for coordinate in location)
        tree.links.new(offset, schedule.inputs["offset"])
        tree.links.new(count, schedule.inputs["count"])
        return schedule

    # ----------------------------------------------------------------
    def _create_grow_life_on_frame(self, tree, control, letters, count, clock):
        """``Grow Life On``: the first word, growing in letter by letter.

        Two ``Delete Geometry`` nodes, both of them the editor's: the first
        drops every outline whose slot has not opened yet (a letter at zero
        size would otherwise leave a bead of tube at its own pivot), the
        second drops every outline that belongs to the other word - which
        after the join is exactly an outline whose ``LattitudeIndex`` is zero.

        :return: ``(geometry, grow)`` - this word's outlines, and how far each
            of them has grown.
        """
        at = _in_frame(self.GROW_FRAME)
        schedule = self._create_letter_clock(
            tree, control, count, at(1.5, -0.5), "LifeLetterClock",
            control["StartTime"].std_out)
        tree.links.new(clock, schedule.inputs["time"])
        tree.links.new(control["TransitionTime"].std_out,
                       schedule.inputs["transition"])
        index = NamedAttribute(tree, location=at(0.1, -0.9), data_type="INT",
                               name=self.LATITUDE, node_height=GRID,
                               label="LattitudeIndex")
        tree.links.new(index.std_out, schedule.inputs["index"])

        waiting = CompareNode(tree, location=at(3.1, -0.9), operation="LESS_THAN",
                              data_type="FLOAT", inputs0=clock,
                              inputs1=schedule.outputs["begin"],
                              node_height=GRID, name="LifeLetterNotDueYet")
        due = DeleteGeometry(tree, location=at(4.3, -1.2), domain="CURVE",
                             mode="ALL", geometry=letters,
                             selection=waiting.std_out, node_height=GRID,
                             name="DropUnstartedLifeLetters")

        belongs = NamedAttribute(tree, location=at(3.0, -0.1), data_type="INT",
                                 name=self.LATITUDE, node_height=GRID,
                                 label="LattitudeIndex")
        foreign = CompareNode(tree, location=at(4.3, -0.2), operation="EQUAL",
                              data_type="INT", inputs0=belongs.std_out,
                              inputs1=0, node_height=GRID, name="NotALifeLetter")
        only = DeleteGeometry(tree, location=at(5.3, -1.2), domain="CURVE",
                              mode="ALL", geometry=due.geometry_out,
                              selection=foreign.std_out, node_height=GRID,
                              name="DropEarthLetters")

        frame = Frame(tree, location=self.GROW_FRAME, label="Grow Life On",
                      node_height=GRID)
        frame.add([schedule, index, waiting, due, belongs, foreign, only])
        return only.geometry_out, schedule.outputs["grow"]

    # ----------------------------------------------------------------
    def _create_earth_frame(self, tree, control, letters, count, clock):
        """``Earth``: the second word, on the same terms one word later.

        Identical to :meth:`_create_grow_life_on_frame` but for the two things
        that make it the second word: its clock starts at ``StartTime +
        TransitionTime`` rather than at ``StartTime``, and it reads and keeps
        ``LongitudeIndex`` rather than ``LattitudeIndex``.

        :return: ``(geometry, grow)``.
        """
        at = _in_frame(self.EARTH_FRAME)
        # StartTime + TransitionTime: `Earth` starts writing itself the
        # moment `Life on` has finished
        after = MathNode(tree, location=at(1.1, -1.4), operation="ADD",
                         inputs0=control["StartTime"].std_out,
                         inputs1=control["TransitionTime"].std_out,
                         node_height=GRID, name="AfterLifeOn")
        schedule = self._create_letter_clock(
            tree, control, count, at(1.4, -0.9), "EarthLetterClock",
            after.std_out)
        tree.links.new(clock, schedule.inputs["time"])
        tree.links.new(control["TransitionTime2"].std_out,
                       schedule.inputs["transition"])
        index = NamedAttribute(tree, location=at(0.1, -0.5), data_type="INT",
                               name=self.LONGITUDE, node_height=GRID,
                               label="LongitudeIndex")
        tree.links.new(index.std_out, schedule.inputs["index"])

        waiting = CompareNode(tree, location=at(3.1, -0.4), operation="LESS_THAN",
                              data_type="FLOAT", inputs0=clock,
                              inputs1=schedule.outputs["begin"],
                              node_height=GRID, name="EarthLetterNotDueYet")
        due = DeleteGeometry(tree, location=at(4.3, -0.1), domain="CURVE",
                             mode="ALL", geometry=letters,
                             selection=waiting.std_out, node_height=GRID,
                             name="DropUnstartedEarthLetters")

        belongs = NamedAttribute(tree, location=at(3.1, -1.3), data_type="INT",
                                 name=self.LONGITUDE, node_height=GRID,
                                 label="LongitudeIndex")
        foreign = CompareNode(tree, location=at(4.3, -0.9), operation="EQUAL",
                              data_type="INT", inputs0=belongs.std_out,
                              inputs1=0, node_height=GRID,
                              name="NotAnEarthLetter")
        only = DeleteGeometry(tree, location=at(5.3, -0.1), domain="CURVE",
                              mode="ALL", geometry=due.geometry_out,
                              selection=foreign.std_out, node_height=GRID,
                              name="DropLifeLetters")

        frame = Frame(tree, location=self.EARTH_FRAME, label="Earth",
                      node_height=GRID)
        frame.add([after, schedule, index, waiting, due, belongs, foreign, only])
        return only.geometry_out, schedule.outputs["grow"]

    # ----------------------------------------------------------------
    def _create_circles_of_longitude_frame(self, tree, control, count, ramp):
        """``Circles Of Longitude``: where a letter of ``Life on`` is going.

        Meridian ``i`` of ``count`` runs at ``phi = (i-1) pi / count`` and is
        drawn by walking ``theta`` once round; the poles land on **y**. The
        whole thing is scaled by ``lambda`` so that at the start of the
        transformation every target sits on the origin, which is what makes
        ``flat + target`` a blend rather than a sum: the flat term is scaled
        by ``1 - lambda`` in the ``Original Position`` frame.

        :return: the vector socket of the target position.
        """
        at = _in_frame(self.MERIDIAN_FRAME)
        latitude = NamedAttribute(tree, location=at(0.1, -0.1), data_type="INT",
                                  name=self.LATITUDE, node_height=GRID,
                                  label="LattitudeIndex")
        sample = Index(tree, location=at(0.4, -1.1), node_height=GRID,
                       name="AlongTheMeridian")
        point = make_function(
            tree, name="MeridianPoint",
            aux_functions={
                "phi": "latIndex,1,-,%.15f,*,count,/" % pi,
                "theta": "index,resolution,%%,%.15f,*,resolution,/" % (2 * pi),
                # not `scale`: that spelling is the RPN vector operator
                "reach": "lam,radius,*",
            },
            functions={"point": ["phi,cos,theta,sin,*,reach,*",
                                 "theta,cos,reach,*",
                                 "phi,sin,theta,sin,*,reach,*"]},
            inputs=["index", "resolution", "count", "latIndex", "radius", "lam"],
            outputs=["point"],
            scalars=["index", "resolution", "count", "latIndex", "radius", "lam",
                     "phi", "theta", "reach"],
            vectors=["point"], hide=True)
        point.location = tuple(coordinate * GRID for coordinate in at(1.5, -0.8))
        tree.links.new(sample.std_out, point.inputs["index"])
        tree.links.new(control["Resolution"].std_out, point.inputs["resolution"])
        tree.links.new(count, point.inputs["count"])
        tree.links.new(latitude.std_out, point.inputs["latIndex"])
        tree.links.new(control["Radius"].std_out, point.inputs["radius"])
        tree.links.new(ramp, point.inputs["lam"])

        frame = Frame(tree, location=self.MERIDIAN_FRAME,
                      label="Circles Of Longitude", node_height=GRID)
        frame.add([latitude, sample, point])
        return point.outputs["point"]

    # ----------------------------------------------------------------
    def _create_circles_of_latitude_frame(self, tree, control, count, ramp):
        """``Circles Of Latitude``: where a letter of ``Earth`` is going.

        The same sphere read the other way round: the letter's index picks the
        *latitude* ``theta = i pi / (count + 1)`` and ``phi`` walks once round
        it. The ``+ 1`` is what keeps the parallels off the poles - the first
        letter would otherwise be a circle of zero radius sitting on the
        north pole, and one letter of the word would simply disappear.

        :return: the vector socket of the target position.
        """
        at = _in_frame(self.PARALLEL_FRAME)
        longitude = NamedAttribute(tree, location=at(0.1, -0.9), data_type="INT",
                                   name=self.LONGITUDE, node_height=GRID,
                                   label="LongitudeIndex")
        sample = Index(tree, location=at(0.2, -0.1), node_height=GRID,
                       name="AroundTheParallel")
        point = make_function(
            tree, name="ParallelPoint",
            aux_functions={
                "phi": "index,resolution,%%,%.15f,*,resolution,/" % (2 * pi),
                "theta": "lonIndex,%.15f,*,count,1,+,/" % pi,
                # not `scale`: that spelling is the RPN vector operator
                "reach": "lam,radius,*",
            },
            functions={"point": ["phi,cos,theta,sin,*,reach,*",
                                 "theta,cos,reach,*",
                                 "phi,sin,theta,sin,*,reach,*"]},
            inputs=["index", "resolution", "count", "lonIndex", "radius", "lam"],
            outputs=["point"],
            scalars=["index", "resolution", "count", "lonIndex", "radius", "lam",
                     "phi", "theta", "reach"],
            vectors=["point"], hide=True)
        point.location = tuple(coordinate * GRID for coordinate in at(1.7, -0.7))
        tree.links.new(sample.std_out, point.inputs["index"])
        tree.links.new(control["Resolution"].std_out, point.inputs["resolution"])
        tree.links.new(count, point.inputs["count"])
        tree.links.new(longitude.std_out, point.inputs["lonIndex"])
        tree.links.new(control["Radius"].std_out, point.inputs["radius"])
        tree.links.new(ramp, point.inputs["lam"])

        frame = Frame(tree, location=self.PARALLEL_FRAME,
                      label="Circles Of Latitude", node_height=GRID)
        frame.add([longitude, sample, point])
        return point.outputs["point"]

    # ----------------------------------------------------------------
    def _create_original_position_frame(self, tree, origin, grow, ramp, tag):
        """``Original Position``: where a point was before the globe took it.

        Two frames of the editor with the same label and the same three
        nodes, one per word, so they are built by one method called twice.

        The flat term is not simply ``Position``: a letter that has not
        finished growing has to be *small*, and small about its own centre
        rather than about the origin of the sentence. So the point is first
        pulled towards its glyph's pivot by however far that letter has got,

            ``pivot + (position - pivot) * grow``

        and only then faded out by ``1 - lambda`` as the globe takes over.
        With ``grow`` at one the first factor is the identity, so once the
        sentence is written this is exactly the editor's ``Position``.

        :param origin: the frame's own location.
        :param tag: ``"Life"`` or ``"Earth"``, to keep the two sets of node
            names apart.
        :return: the vector socket of the flat contribution.
        """
        at = _in_frame(origin)
        where = Position(tree, location=at(0.1, -0.1), node_height=GRID,
                         name="%sPointPosition" % tag)
        pivot = NamedAttribute(tree, location=at(0.1, -0.9),
                               data_type="FLOAT_VECTOR", name=self.PIVOT,
                               node_height=GRID, label="GlyphPivot")
        flat = make_function(
            tree, name="%sFlatPosition" % tag,
            functions={"flat": [
                "pivot_x,pos_x,pivot_x,-,grow,*,+,1,lam,-,*",
                "pivot_y,pos_y,pivot_y,-,grow,*,+,1,lam,-,*",
                "pivot_z,pos_z,pivot_z,-,grow,*,+,1,lam,-,*"]},
            inputs=["pos", "pivot", "grow", "lam"], outputs=["flat"],
            vectors=["pos", "pivot", "flat"], scalars=["grow", "lam"], hide=True)
        flat.location = tuple(coordinate * GRID for coordinate in at(1.2, -0.4))
        tree.links.new(where.std_out, flat.inputs["pos"])
        tree.links.new(pivot.std_out, flat.inputs["pivot"])
        tree.links.new(grow, flat.inputs["grow"])
        tree.links.new(ramp, flat.inputs["lam"])

        frame = Frame(tree, location=origin, label="Original Position",
                      node_height=GRID)
        frame.add([where, pivot, flat])
        return flat.outputs["flat"]

    # ----------------------------------------------------------------
    def _create_globe_frame(self, tree, control):
        """``Globe``: the ball the graticule ends up drawn on.

        It swells from nothing to ``Radius - TubeRadius`` over the last
        ``GlobeLead`` seconds of the transformation. Two things are being
        bought with that timing and that radius: the ball arrives while the
        letters are still moving, so it reads as the thing they were heading
        for rather than as something switched on afterwards; and its surface
        lands exactly on the *inside* of the tubes, which is close enough to
        look like one object and far enough that nothing z-fights.

        How much of an arrival it is, is what ``GlobeLead`` decides. At the
        editor's 0.1 s it is six frames - the ball snaps in under a graticule
        that is already almost home, which is the reading where the letters
        are the subject and the ocean is what they turn out to have been
        drawn on. Give it half a second and it inflates instead.

        :return: the geometry socket of the ball.
        """
        at = _in_frame(self.GLOBE_FRAME)
        clock = SceneTime(tree, location=at(0.1, -0.2), std_out="Seconds",
                          node_height=GRID, hide=True, name="GlobeClock")
        swell = make_function(
            tree, name="GlobeRadius",
            functions={"radius": "R,tube,-,time,transformTime,duration,+,"
                                 "lead,-,-,lead,/,0,max,1,min,*"},
            inputs=["time", "transformTime", "duration", "lead", "R", "tube"],
            outputs=["radius"],
            scalars=["time", "transformTime", "duration", "lead", "R", "tube",
                     "radius"], hide=True)
        swell.location = tuple(coordinate * GRID for coordinate in at(1.0, -0.1))
        tree.links.new(clock.std_out, swell.inputs["time"])
        tree.links.new(control["TransformTime"].std_out, swell.inputs["transformTime"])
        tree.links.new(control["TransformDuration"].std_out, swell.inputs["duration"])
        tree.links.new(control["GlobeLead"].std_out, swell.inputs["lead"])
        tree.links.new(control["Radius"].std_out, swell.inputs["R"])
        tree.links.new(control["TubeRadius"].std_out, swell.inputs["tube"])

        ball = UVSphere(tree, location=at(1.9, -0.3), radius=swell.outputs["radius"],
                        segments=self.globe_segments, rings=self.globe_rings,
                        node_height=GRID, name="Globe")
        smooth = SetShadeSmooth(tree, location=at(2.9, -0.3),
                                geometry=ball.geometry_out, node_height=GRID,
                                name="SmoothGlobe")
        painted = SetMaterial(tree, location=at(3.8, -0.6),
                              geometry=smooth.geometry_out,
                              material=control["GlobeMaterial"].std_out,
                              node_height=GRID, name="PaintGlobe")

        frame = Frame(tree, location=self.GLOBE_FRAME, label="Globe",
                      node_height=GRID)
        frame.add([clock, swell, ball, smooth, painted])
        return painted.geometry_out

    # ----------------------------------------------------------------
    def _create_curve_geometry_frame(self, tree, control, life, earth):
        """``CurveGeometry``: the two words joined and swept into tube.

        The last thing that happens to the lettering, and the first that
        treats it as one object: both halves have been moved onto the sphere
        by now, so what joins here is a single set of closed curves, whether
        they are still spelling the sentence or already drawing the graticule.

        The editor sweeps them with the "Curve to Tube" asset group; a circle
        and a ``Curve to Mesh`` is the same thing out of nodes this file
        already has, and it does not depend on an asset library being linked
        into the .blend.

        :return: the geometry socket of the painted tubes.
        """
        at = _in_frame(self.TUBE_FRAME)
        grid = JoinGeometry(tree, location=at(0.1, -0.5), node_height=GRID,
                            name="JoinGraticule")
        tree.links.new(life, grid.geometry_in)
        tree.links.new(earth, grid.geometry_in)

        profile = CurveCircle(tree, location=at(0.1, -0.9),
                              resolution=self.profile_resolution,
                              radius=control["TubeRadius"].std_out,
                              node_height=GRID, name="TubeProfile")
        tube = CurveToMesh(tree, location=at(1.0, -0.1), curve=grid.geometry_out,
                           profile_curve=profile.geometry_out, fill_caps=True,
                           node_height=GRID, name="CurveToTube")
        smooth = SetShadeSmooth(tree, location=at(1.9, -0.4),
                                geometry=tube.geometry_out, node_height=GRID,
                                name="SmoothTube")
        painted = SetMaterial(tree, location=at(2.8, -0.4),
                              geometry=smooth.geometry_out,
                              material=control["LetterMaterial"].std_out,
                              node_height=GRID, name="PaintLetters")

        frame = Frame(tree, location=self.TUBE_FRAME, label="CurveGeometry",
                      node_height=GRID)
        frame.add([grid, profile, tube, smooth, painted])
        return painted.geometry_out


class EpochCounterModifier(GeometryNodesModifier):
    """``Epoch: n``, counting up ten a frame.

    The read-out the soup runs behind: a word and a number, and the number is
    the frame the scene is on rather than anything keyframed. So it costs one
    ``Scene Time`` and cannot fall out of step with a render that skips
    frames - see :meth:`BrainFuckSimpleModifier.create_node` for the other
    side of that argument, where a simulation zone *does* need the frames one
    at a time.

    The count is ``(frame - StartFrame) * Step``, held at zero before the
    scene starts and at ``LastEpoch`` once it is over, so the shot can be held
    on the finished number for as long as the cut needs. It reaches
    :attr:`epochs` at :attr:`frames` - ask the scene-side for both rather
    than working them out twice.

    ``Value to String`` is what makes this a geometry-nodes job rather than a
    text object: the number is not known when the tree is built, so the glyphs
    have to be chosen while it runs. Everything after it is the usual way of
    turning a string into something a camera can see - outlines to curves,
    curves realized and filled, and one material over the lot; see
    :meth:`_lettering`, which is that line of nodes.

    **The word and the number are two lines of lettering, not one string.**
    Joined into one and centred, the pair would slide left every time the
    number gained a digit, and the word would never be twice in the same
    place. So the number is centred on the origin, where it grows about its
    own middle, and the word is set once at :attr:`label_offset` and does not
    move again. It costs a second ``String to Curves`` and a ``Join
    Geometry``, and it is the only way the read-out holds still.

    :param step: how far the count goes up from one frame to the next
    :param last_epoch: the number it stops at
    :param label: what stands in front of the number, its trailing space and
        all
    :param text_size: cap height of the number, in blender units
    :param label_size: cap height of the word, which is set on its own rather
        than fed from ``TextSize``: the word is a caption and stays the size
        it is written at whatever the number is scaled to
    :param label_offset: where the word stands, in x, from the middle of the
        number
    :param start_frame: the frame the count leaves zero on
    :param color: the colour of the lettering; ``emission`` and the rest of
        the material's keywords come through ``kwargs``
    """

    def __init__(self, step=10, first_epoch=0,last_epoch=10000, label="Epoch: ",
                 text_size=1.0, label_size=1.0, label_offset=-2.7,
                 start_frame=1, frame_skip=10, color="text",
                 name="EpochCounter", **kwargs):
        self.frame_skip = frame_skip
        self.step = step
        self.first_epoch = first_epoch
        self.last_epoch = last_epoch
        self.label = label
        self.text_size = text_size
        self.label_size = label_size
        self.label_offset = label_offset
        self.start_frame = start_frame
        self.color = color
        self.kwargs = kwargs
        super().__init__(name=name, automatic_layout=False)

    # ----------------------------------------------------------------
    @property
    def frames(self):
        """How many frames the count takes to arrive at :attr:`last_epoch`."""
        return int(math.ceil(self.last_epoch / self.step * self.frame_skip))

    @property
    def duration(self):
        """... and how many seconds that is, for the scene to hold on."""
        return self.frames / FRAME_RATE

    # ----------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        control = {
            "Step": InputValue(tree, location=(0, 0), value=self.step,
                               name="Step"),
            "FrameSkip": InputInteger(tree, location=(0, 0), integer=self.frame_skip, name="FrameSkip"),
            "FirstEpoch": InputInteger(tree, location=(0, -0.8),integer=self.first_epoch, name="FirstEpoch"),
            "LastEpoch": InputInteger(tree, location=(0, -0.8),integer=self.last_epoch, name="LastEpoch"),
            "StartFrame": InputValue(tree, location=(0, -1.6),
                                     value=self.start_frame, name="StartFrame"),
            "TextSize": InputValue(tree, location=(0, -2.4), value=self.text_size,
                                   name="TextSize"),
        }
        colour = InputMaterial(tree, location=(0, -3.2), material=self.color,
                               name="EpochColor", **self.kwargs)
        self.materials.append(colour.node.material)
        frame = Frame(tree, location=(-0.4, 0.8), label="ControlParameter")
        frame.add(list(control.values()) + [colour])

        now = SceneTime(tree, location=(2, 0), std_out="Frame", name="EpochClock")
        # held at zero before the scene starts and at LastEpoch after it ends,
        # so that the number is right on a frame either side of the count
        count = make_function(
            tree, name="EpochNumber", location=(3.4, -0.4), hide=False,
            functions={"Epoch": "frame,frameSkip,/,floor,start,-,0,max,step,*,last,min,first,+"},
            inputs=["frame", "start", "step", "first","last", "frameSkip"], outputs=["Epoch"],
            scalars=["step"], integers=["start", "frameSkip", "frame", "first","last", "Epoch"])
        for socket, socket_name in ((now.std_out, "frame"),
                                    (control["StartFrame"].std_out, "start"),
                                    (control["Step"].std_out, "step"),
                                    (control["FirstEpoch"].std_out, "first"),
                                    (control["LastEpoch"].std_out, "last"),
                                    (control["FrameSkip"].std_out, "frameSkip")):
            tree.links.new(socket, count.inputs[socket_name])
        digits = ValueToString(tree, location=(5, -0.4), data_type="INT",
                               value=count.outputs["Epoch"], name="EpochDigits")
        word = InputString(tree, location=(5.1, 1.4), string=self.label,
                           name="EpochLabel")

        # the number, centred on the origin, and the word, held to the left of
        # it - two lines rather than one string, see the class docstring
        number = self._lettering(tree, digits.std_out,
                                 control["TextSize"].std_out, colour.std_out,
                                 origin=(7.8, 0))
        caption = self._lettering(tree, word.std_out, self.label_size,
                                  colour.std_out, origin=(7.8, 2.8),
                                  translation=[self.label_offset, 0, 0])
        joined = JoinGeometry(tree, location=(13, 0.6), hide=True,
                              geometry=[number[-1].geometry_out,
                                        caption[-1].geometry_out])

        counter = Frame(tree, location=(1.6, 0.8), label="EpochCounter")
        counter.add([now, count, word, digits] + number + caption)

        self.group_outputs.location = (14.1 * 200, 0.3 * 200)
        tree.links.new(joined.geometry_out, self.group_outputs.inputs["Geometry"])

    # ----------------------------------------------------------------
    @staticmethod
    def _lettering(tree, string, size, material, origin, translation=None):
        """One line of text: a string socket in, something a camera can see out.

        Outlines to curves, curves realized and filled, one material over the
        lot, and a quarter turn to stand the result up - the same five nodes
        for the word as for the number, which is why they are written once.

        :param string: the socket the text comes from
        :param size: cap height, a socket or a number
        :param material: the socket every glyph is painted from
        :param origin: where the line of nodes starts in the editor
        :param translation: where the finished lettering stands, or ``None``
            to leave it on the origin
        :return: the nodes of the line, the last of them carrying the geometry
        """
        x, y = origin
        curves = StringToCurves(tree, location=(x, y), string=string, size=size,
                                align_x="CENTER", align_y="MIDDLE",
                                name="EpochCurves")
        realize = RealizeInstances(tree, location=(x + 1.2, y))
        fill = FillCurve(tree, location=(x + 2.2, y), mode="N-gons")
        painted = SetMaterial(tree, location=(x + 3.2, y), material=material,
                              name="PaintEpoch")
        # String to Curves writes in the x-y plane; a quarter turn about x
        # stands the words up for a camera looking along +y
        stood = TransformGeometry(tree, location=(x + 4.2, y),
                                  translation=[0, 0, 0] if translation is None
                                  else translation,
                                  rotation=[pi / 2, 0, 0], name="StandEpochUp")
        create_geometry_line(tree, [realize, fill, painted, stood],
                             ins=curves.geometry_out)
        return [curves, realize, fill, painted, stood]


class BrainFuckHelloModifier(BrainFuckExtendedModifier):
    """The two-headed machine writing HELLO onto one short tape.

    :class:`BrainFuckExtendedModifier` runs a soup: two tapes of sixty-four
    bytes from csv, and a program that is whatever those bytes happen to be.
    This is the same machine with the soup taken away - one tape, short enough
    to read, holding one program that was written on purpose::

        {{{{{{++++[>++<-]>.}>+++++.}<++++.}.}+++.}

    and it is the argument of the whole video in one shot: *the same HELLO*
    the one-headed machine printed, on a machine that cannot print. There is
    no output box here, because there is no output - a BFF program says what
    it has to say by writing it onto the tape, so HELLO has to appear on the
    tape itself, and that is what the second head is for.

    **How the tape is laid out.** Three kinds of cell, and the program needs
    all three:

    ``0`` to ``scratch - 1``
        zero, and the counter walks over them as no-ops on its way in. They
        are head0's workspace: ``++++[>++<-]`` counts eight into the second of
        them and the later runs add to it, so this is where the numbers are
        built.
    ``scratch`` on
        the program, one character per cell as its ascii code. The counter
        reads a cell and takes its value for an opcode, so this *is* the
        program - and it draws as one, because a cell holding the code of an
        instruction shows the instruction.
    the last ``spare``
        zero, and where the answer lands. ``{`` walks head1 *left* off cell 0
        and round the ring onto them, which is why they are at the far end and
        why there have to be exactly six: one per letter, and one more for the
        head to finish on.

    **Why fifty-one cells.** Three, forty-two and six. One fewer and head1
    comes round the ring into the program and overwrites it; one more and the
    letters land a cell further along with a gap behind them. The machine
    halts after 69 steps, when the counter walks off the end.

    **Why the cells read as letters.** The values written are 8, 5, 12, 12 and
    15 - the alphabet encoding of the one-headed machine, where ``A`` is 1 and
    ``Z`` is 26, and the reason that machine could print HELLO in 27
    instructions. The extended machine draws a cell as its number unless the
    number is the code of an instruction, so as it stands the answer would
    read ``8 5 12 12 15``. :meth:`_cell_glyph` puts a third case in between:
    a value from 1 to 26 draws as the letter of :attr:`LETTERS` it stands for.
    Set ``letters=False`` to see the numbers instead, which is the same tape
    read the machine's own way.

    :param program: the BFF program, written onto the tape from cell
        ``scratch``
    :param scratch: zero cells in front of it for head0 to work in
    :param spare: cells behind it for head1 to write the answer into
    :param letters: read the answer back through the table above the tape
        once the machine has halted, rather than leaving it as numbers
    """

    #: HELLO on the two-headed machine. Five ``{`` put head1 on the last five
    #: cells of the ring - one per letter, with nothing left over - and each
    #: ``}`` walks it one to the right for the next; the arithmetic between
    #: them is the one-headed HELLO with its prints turned into copies. The
    #: five ``}`` bring head1 round to cell 0 again, so it ends where it
    #: started and no cell is spent on parking it.
    HELLO = "{{{{{++++[>++<-]>.}>+++++.}<++++.}.}+++.}"

    #: What the two cells head0 adds up in start out holding, and the whole
    #: trick of this machine: 64 is where the capitals begin in ascii, so
    #: ``++++[>++<-]`` counting eight into a cell that holds 64 leaves 72 -
    #: and 72 *is* ``H``. So the machine writes ascii rather than a code that
    #: has to be translated afterwards, and the table above the tape is the
    #: right table to read its answer in. Cell 0 stays at zero: it is the
    #: loop's counter and the loop ends when it runs out.
    LETTER_ORIGIN = 64

    #: The cursor is a rectangle rather than a square here, tall enough to
    #: take in the character standing on the cell as well as the cell itself -
    #: an instruction is drawn ``cell_command_scale`` times the size of a
    #: number, and it is the instruction the cursor is pointing out.
    cursor_tall = 2.6
    cursor_lift = 0.45

    def __init__(self, program=None, scratch=3, gap=1, spare=5, letters=True,
                 cell_size=0.5, name="HelloExtended", **kwargs):
        self.bff_program = self.HELLO if program is None else program
        self.scratch = scratch
        self.gap = gap
        self.spare = spare
        self.letters = letters
        # one tape, and no csv files to fill it from
        kwargs.pop("tape_files", None)
        super().__init__(
            tape_size=scratch + len(self.bff_program) + gap + spare,
            cell_size=cell_size, tape_files=(), name=name, **kwargs)

    # ----------------------------------------------------------------
    @property
    def first_answer(self):
        """The first cell the answer is written on."""
        return self.scratch + len(self.bff_program) + self.gap

    @property
    def first_instruction(self):
        """Where the counter starts: the first cell that is not zero.

        The scratch cells in front of the program are zero, and a zero is a
        no-op, so a counter starting at 0 spends its first steps walking over
        nothing. It starts on the program instead, which is also where the
        cursor first appears.
        """
        return self.scratch

    @property
    def halt_at(self):
        """Where the counter stops: the zero cell after the last instruction.

        This machine knows where its program ends, so it does not have to run
        off the end of memory to find out - it stops on the zero that was put
        there for it. Everything past that point is the answer, and reading
        the answer as instructions is what the soup does, not what this does.
        """
        return self.scratch + len(self.bff_program)

    # ----------------------------------------------------------------
    @property
    def steps(self):
        """How many instructions the machine executes before it halts.

        The counter starts at 0 and moves one cell at a time except where a
        bracket sends it back, so this is worked out by running the thing -
        see :meth:`simulate`. The scene needs it to know how long the shot is.
        """
        return self.simulate()[0]

    def simulate(self):
        """Run the program in python, exactly as the graph runs it.

        :return: ``(steps, tape)`` - the tape as the machine leaves it.
        """
        tape = ([0] + [self.LETTER_ORIGIN] * (self.scratch - 1)
                + [ord(character) for character in self.bff_program]
                + [0] * (self.gap + self.spare))
        size = len(tape)
        head0 = head1 = steps = 0
        counter = self.first_instruction
        while self.first_instruction <= counter < self.halt_at:
            byte = tape[counter]
            code = chr(byte) if 32 <= byte < 127 else ""
            onward = counter + 1
            if code == ">":
                head0 = (head0 + 1) % size
            elif code == "<":
                head0 = (head0 - 1) % size
            elif code == "}":
                head1 = (head1 + 1) % size
            elif code == "{":
                head1 = (head1 - 1) % size
            elif code == "+":
                tape[head0] = (tape[head0] + 1) % 256
            elif code == "-":
                tape[head0] = (tape[head0] - 1) % 256
            elif code == ".":
                tape[head1] = tape[head0]
            elif code == ",":
                tape[head0] = tape[head1]
            elif code in "[]":
                # the partner is searched for in the tape as it stands, the
                # way _create_bracket_scan does it
                forward, depth = code == "[", 1
                jumps = (byte == ord("[") and tape[head0] == 0) or \
                        (byte == ord("]") and tape[head0] != 0)
                if jumps:
                    index = counter + (1 if forward else -1)
                    while 0 <= index < size and depth:
                        here = chr(tape[index]) if 32 <= tape[index] < 127 else ""
                        depth += (here == code) - (here == ("]" if forward else "["))
                        index += 1 if forward else -1
                    onward = index if depth == 0 else size
                    if not forward and depth == 0:
                        onward = index + 2
            counter, steps = onward, steps + 1
        return steps, tape

    # ----------------------------------------------------------------
    def _create_tape_frame(self, tree, control):
        """``Tape``: one tape, with the program written into the middle of it.

        No csv and no second tape. The value of a cell is the ascii code of
        the program character that belongs on it and zero everywhere else,
        which is the whole of the layout the class docstring describes - a
        ``Slice String`` on the program and a range test around it.

        :return: the geometry socket of the initial tape.
        """
        program = InputString(tree, location=(-8, -1.4), string=self.bff_program,
                              name="Program")
        length = MathNode(tree, location=(-8, 0), operation="MULTIPLY",
                          inputs0=control["TapeSize"].std_out,
                          inputs1=control["CellSize"].std_out, name="TapeLength")
        end = CombineXYZ(tree, location=(-7, 0), x=length.std_out, name="TapeEnd")
        line = MeshLine(tree, location=(-6, 0.6), mode="END_POINTS",
                        count=control["TapeSize"].std_out,
                        start_location=Vector([0, 0, 0]), end_location=end.std_out)

        cell = Index(tree, location=(-8, -2.0), name="CellIndex")
        where = make_function(
            tree, name="ProgramCell", location=(-6.6, -1.7), hide=True,
            aux_functions={"at": "i,%d,-" % self.scratch},
            functions={"At": "at",
                       "OnIt": "at,0,<,not,at,%d,<,and" % len(self.bff_program),
                       # the cells head0 adds up in start at the origin; cell
                       # 0 is the loop counter, and everything past the
                       # program - the zero it ends on and the cells the
                       # answer lands on - starts at nothing
                       "Based": "i,0,>,i,%d,<,and" % self.scratch},
            inputs=["i"], outputs=["At", "OnIt", "Based"],
            integers=["i", "at", "At"], booleans=["OnIt", "Based"])
        tree.links.new(cell.std_out, where.inputs["i"])
        character = SliceString(tree, location=(-5.4, -1.4), string=program.std_out,
                                position=where.outputs["At"], length=1,
                                name="ProgramCharacter")
        code = CharToAscii(tree, location=(-4.6, -1.4), char=character.std_out,
                           name="ProgramByte")
        empty = Switch(tree, location=(-4.6, -2.2), input_type="INT",
                       switch=where.outputs["Based"], false=0,
                       true=self.LETTER_ORIGIN, name="BlankOrOrigin")
        byte = Switch(tree, location=(-3.8, -1.4), input_type="INT",
                      switch=where.outputs["OnIt"], false=empty.std_out,
                      true=code.std_out, name="CellByte")
        # the attribute has to exist from the first frame on, otherwise the
        # Sample Index in the automaton has nothing to read
        values = StoredNamedAttribute(tree, location=(-3.0, 0.6), data_type="INT",
                                      domain="POINT", name="Value",
                                      value=byte.std_out, label="LoadTape")
        # one tape, so every cell is on tape 0 and its number is its index -
        # the Cells frame drops each cell onto the line of its own tape and
        # the arrows read the cell number back off the realized geometry
        tape_kind = StoredNamedAttribute(tree, location=(-2.2, 0.6), data_type="INT",
                                         domain="POINT", name="Tape", value=0,
                                         label="TapeNumber")
        number = StoredNamedAttribute(tree, location=(-1.4, 0.6), data_type="INT",
                                      domain="POINT", name="Cell",
                                      value=cell.std_out, label="CellNumber")
        create_geometry_line(tree, [line, values, tape_kind, number])

        frame = Frame(tree, location=(-8.4, 1.4), label="Tape")
        frame.add([program, length, end, line, cell, where, character, code,
                   empty, byte, values, tape_kind, number])
        return number.geometry_out

    # ----------------------------------------------------------------
    def _cell_glyph(self, tree, control, held, digits, letter, is_command,
                    cell=None, counter=None, location=(0, 0)):
        """An instruction, a number, or - at the very end - a letter.

        The cells the answer lands on start at nothing and show it, and they
        show what is copied onto them as it arrives, which is 72 and 69 and
        76: numbers, because a number is what a cell holds. Only once the
        machine has halted are they read as what the table above the tape
        says they are, and then the tape says HELLO.

        Nothing is translated to do that. The 64 the two working cells start
        at has already done the work - the machine wrote ascii - so this is
        the same ``Slice String`` on ``CodeTable`` that draws an instruction,
        asked for a few cells more.
        """
        if not self.letters:
            return super()._cell_glyph(tree, control, held, digits, letter,
                                       is_command, cell=cell, counter=counter,
                                       location=location)
        x, y = location
        reading = make_function(
            tree, name="CellReading", location=(x - 1.4, y - 1.2), hide=True,
            # an instruction always; a cell of the answer once the machine has
            # stopped, and only if what it holds is something the table shows
            functions={"AsLetter": "command,counter,%d,<,not,i,%d,<,not,and,"
                                   "value,31,<,not,and,value,127,<,and,or"
                                   % (self.halt_at, self.first_answer)},
            inputs=["value", "i", "counter", "command"], outputs=["AsLetter"],
            integers=["value", "i", "counter"],
            booleans=["command", "AsLetter"])
        for socket, socket_name in ((held, "value"), (cell, "i"),
                                    (counter, "counter"),
                                    (is_command, "command")):
            tree.links.new(socket, reading.inputs[socket_name])
        return Switch(tree, location=(x, y), input_type="STRING",
                      switch=reading.outputs["AsLetter"], true=letter,
                      false=digits, name="LetterOrNumber").std_out
