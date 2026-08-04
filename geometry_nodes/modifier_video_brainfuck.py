import math
import os

import numpy as np

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
    CurveLength, SeparateGeometry, InputRotation, MorphNode
from interface.ibpy import Vector
from objects.logo import logo_curve
from utils.constants import DATA_DIR
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
        ``Step``).

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
        instructions again. Each column is coloured by what has become of it,
        so that what is dark behind the box is what has run for the last time
        and what is not is waiting for the next turn of its loop.

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
        :attr:`CELL_COLORS` and :attr:`PROGRAM_COLORS`, and the two entries
        ``GlyphColor`` and ``FrameColor``
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
        ("PointerColor", "important"),  # the cell the head is on
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

    # ascii codes of the seven instructions
    DOT, PLUS, MINUS, LEFT, RIGHT = ord("."), ord("+"), ord("-"), ord("<"), ord(">")
    OPEN, CLOSE = ord("["), ord("]")

    # The jump table is carried as a string, one character per instruction, so
    # that it can be read with the same Slice String the instruction is read
    # with. Destinations are offset by this so that they stay printable - the
    # lookup in Char To Ascii only covers codes 32 to 126, and 0 is a null
    # byte rather than a character in the first place.
    JUMP_ORIGIN = ord("0")

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

        cells = self._create_cells_frame(tree, control, run)
        table = self._create_code_table_frame(tree, control)
        displays = [
            self._create_display_frame(tree, control, "InputDisplay",
                                       control["InputDisplaySize"],
                                       control["InputPosition"], location=(26, -21)),
            self._create_display_frame(tree, control, "OutputDisplay",
                                       control["OutputDisplaySize"],
                                       control["OutputPosition"], location=(26, -24)),
        ]
        simulated = self._create_simulated_geometry_frame(tree, control, variables, run)

        out = self.group_outputs
        out.location = (38 * 200, -2 * 200)
        join = JoinGeometry(tree, location=(36, -4))
        for piece in [cells, table, simulated] + displays:
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

        # one Input Material node per colour of a cell, plus the two that
        # everything else is drawn in
        palette = {}
        for row, (node_name, color) in enumerate(self.cell_colors):
            palette[node_name] = InputMaterial(tree, location=(x, -4.4 - 0.4 * row),
                                               material=color, name=node_name,
                                               **self.kwargs)
        rest = [("GlyphColor", self.glyph_color),
                ("FrameColor", self.frame_color)]
        for offset, (node_name, color) in enumerate(rest):
            palette[node_name] = InputMaterial(
                tree, location=(x, -4.4 - 0.4 * (len(self.cell_colors) + offset)),
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
                ("OutputPosition", [middle, 0, below]))):
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

        frame = Frame(tree, location=(-24, 0.6), label="ControlParameter")
        frame.add(list(control.values()))
        return control

    # ----------------------------------------------------------------
    def _create_variables_frame(self, tree):
        """``Variables``: the program, its jump table and the four state seeds."""
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
        frame = Frame(tree, location=(-16, 0.6), label="Variables")
        frame.add(list(variables.values()))
        return variables

    # ----------------------------------------------------------------
    def _create_tape_frame(self, tree, control):
        """``Tape``: the cells as the machine starts them, all holding zero.

        :return: the geometry socket of the initial tape.
        """
        length = MathNode(tree, location=(-8, 0), operation="MULTIPLY",
                          inputs0=control["TapeSize"].std_out,
                          inputs1=control["CellSize"].std_out, name="TapeLength")
        end = CombineXYZ(tree, location=(-7, 0), x=length.std_out, name="TapeEnd")
        line = MeshLine(tree, location=(-6, 0.6), mode="END_POINTS",
                        count=control["TapeSize"].std_out,
                        start_location=Vector([0, 0, 0]), end_location=end.std_out)
        # every cell starts empty. The attribute has to exist from the first
        # frame on, otherwise the "Sample Index" in the automaton has nothing
        # to read and the cells have nothing to be coloured by.
        zeros = StoredNamedAttribute(tree, location=(-4.6, 0.6), data_type="INT",
                                     domain="POINT", name="Value", value=0,
                                     label="ClearTape")
        create_geometry_line(tree, [line, zeros])
        frame = Frame(tree, location=(-8.2, 1.4), label="Tape")
        frame.add([length, end, line, zeros])
        return zeros.geometry_out

    # ----------------------------------------------------------------
    def _create_run_program_frame(self, tree, control, variables, tape):
        """``RunProgram``: the simulation zone - the clock and the program counter.

        :return: ``{name: socket}`` of the state as it leaves the zone.
        """
        zone = Simulation(tree, location=(2, 5), node_width=20, geometry=tape)
        sim_in, sim_out = zone.simulation_input, zone.simulation_output
        for socket_type, socket_name, initial in (
                ("FLOAT", "StartTime", control["StartTime"].std_out),
                ("INT", "Step", variables["Step"].std_out),
                ("INT", "PointerPosition", variables["Pointer"].std_out),
                ("INT", "Counter", variables["Counter"].std_out),
                ("STRING", "Output", variables["Output"].std_out),
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

        # --- the instruction under the counter --------------------------
        program = variables["Input"].std_out
        current = SliceString(tree, location=(3.2, 4.6), string=program,
                              position=sim_in.outputs["Counter"], length=1,
                              name="Instruction")
        opcode = CharToAscii(tree, location=(4.4, 4.6), char=current.std_out)
        length = StringLength(tree, location=(3.2, 3.6), string=program,
                              name="ProgramLength")
        # the clock keeps going after the last instruction, so without this the
        # machine would go on "executing" the empty slice past the end of the
        # program: the head would stay put but the read-out would blank and the
        # tape would keep being rewritten. Halting when the counter runs off
        # the end leaves the finished state up.
        running = CompareNode(tree, location=(4.4, 3.6), operation="LESS_THAN",
                              data_type="INT", inputs0=sim_in.outputs["Counter"],
                              inputs1=length.std_out, name="NotHalted")
        fire = BooleanMath(tree, location=(10.4, 6.4), operation="AND",
                           inputs0=advance.std_out, inputs1=running.std_out,
                           name="ExecuteNow")

        # --- the reroutes that carry the decoded step into the automaton ---
        code_in = Reroute(tree, location=(11.6, 4.6), ins=opcode.std_out, name="Opcode")
        fire_in = Reroute(tree, location=(11.6, 4.2), ins=fire.std_out, name="Fire")
        head_in = Reroute(tree, location=(11.6, 3.8),
                          ins=sim_in.outputs["PointerPosition"], name="Head")
        step_in = Reroute(tree, location=(11.6, 3.4), ins=sim_in.outputs["Counter"],
                          name="Counter")

        pointer, tape_out, output, counter = self._create_automaton_frame(
            tree, control, variables, sim_in, code_in.std_out, fire_in.std_out,
            head_in.std_out, step_in.std_out)

        for socket, name in ((time.std_out, "Time"), (step.std_out, "Step"),
                             (sim_in.outputs["StartTime"], "StartTime"),
                             (counter, "Counter"),
                             (pointer, "PointerPosition"), (output, "Output")):
            tree.links.new(socket, sim_out.inputs[name])
        # replaces the pass-through that the Simulation wrapper puts in
        tree.links.new(tape_out, sim_out.inputs["Geometry"])

        frame = Frame(tree, location=(1.6, 7.4), label="RunProgram")
        frame.add([zone, time, since, scaled, waiting, step, advance, running, fire,
                   current, opcode, length,
                   code_in, fire_in, head_in, step_in])
        return {name: sim_out.outputs[name] for name in
                ("Geometry", "Step", "PointerPosition", "Counter", "Output")}

    # ----------------------------------------------------------------
    def _create_automaton_frame(self, tree, control, variables, sim_in, opcode,
                                fire, head, counter):
        """``Automaton``: what the seven instructions do.

        Every instruction is one ``Compare`` against its ascii code, ``AND``-ed
        with *fire* - "a new step has just begun". Without that ``AND`` an
        instruction would take effect once per rendered frame instead of once.

        :param counter: the program counter as the frame starts
        :return: ``(pointer, tape, output, counter)`` sockets for the state to
            be written back into the simulation.
        """
        built = []

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
        forward = keep(IntegerMath(tree, location=(16.0, 2.8), operation="ADD",
                                   inputs0=head, inputs1=right, name="HeadRight"))
        backward = keep(IntegerMath(tree, location=(17.0, 2.8), operation="SUBTRACT",
                                    inputs0=forward.std_out, inputs1=left,
                                    name="HeadLeft"))
        # the tape does not wrap and it does not grow, so the head is kept on
        # it - without this a program with one ">" too many would write into a
        # cell that is not drawn and silently do nothing visible
        last = keep(IntegerMath(tree, location=(16.0, 2.0), operation="SUBTRACT",
                                inputs0=control["TapeSize"].std_out, inputs1=1,
                                name="LastCell"))
        capped = keep(IntegerMath(tree, location=(18.0, 2.8), operation="MINIMUM",
                                  inputs0=backward.std_out, inputs1=last.std_out,
                                  name="NotPastTheEnd"))
        pointer = keep(IntegerMath(tree, location=(19.0, 2.8), operation="MAXIMUM",
                                   inputs0=capped.std_out, inputs1=0,
                                   name="NotBeforeStart"))

        # --- the cell under the head ------------------------------------
        # this is where the values live: an integer attribute of the tape
        # geometry, which the simulation zone hands from frame to frame
        stored = NamedAttribute(tree, location=(12.4, 0.2), data_type="INT",
                                name="Value")
        cell = SampleIndex(tree, location=(13.6, 0.2), data_type="INT", domain="POINT",
                           geometry=sim_in.outputs["Geometry"], value=stored.std_out,
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
        tree.links.new(sim_in.outputs["Geometry"], tape.geometry_in)

        # --- printing ---------------------------------------------------
        # the point of the exercise: the cell value indexes into the code
        # table, so 8 prints H. The table is 1-based, hence the -1.
        place = IntegerMath(tree, location=(13.6, 4.8), operation="SUBTRACT",
                            inputs0=cell.std_out, inputs1=1, name="TableIndex")
        letter = SliceString(tree, location=(14.8, 4.8), string=control["CodeTable"].std_out,
                             position=place.std_out, length=1, name="Letter")
        # an empty cell has no letter; without this "." on a zero cell would
        # print an A, because slicing at -1 clamps to the front of the table
        holds = CompareNode(tree, location=(13.6, 5.6), operation="GREATER_THAN",
                            data_type="INT", inputs0=cell.std_out, inputs1=0,
                            name="CellHoldsALetter", hide=True)
        prints = BooleanMath(tree, location=(14.8, 5.6), operation="AND",
                             inputs0=decodes(self.DOT, 6.4, "Dot"), inputs1=holds.std_out,
                             name="DoPrint", hide=True)
        printed = Switch(tree, location=(16.0, 5.6), input_type="STRING",
                         switch=prints.std_out, false="", true=letter.std_out,
                         name="Printed")
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
        empty = CompareNode(tree, location=(14.8, -1.6), operation="EQUAL",
                            data_type="INT", inputs0=cell.std_out, inputs1=0,
                            name="CellIsEmpty", hide=True)
        filled = CompareNode(tree, location=(14.8, -2.4), operation="NOT_EQUAL",
                             data_type="INT", inputs0=cell.std_out, inputs1=0,
                             name="CellIsNotEmpty", hide=True)
        skips = BooleanMath(tree, location=(16.0, -1.6), operation="AND",
                            inputs0=decodes(self.OPEN, -1.6, "Open"),
                            inputs1=empty.std_out, name="SkipLoop", hide=True)
        repeats = BooleanMath(tree, location=(16.0, -2.4), operation="AND",
                              inputs0=decodes(self.CLOSE, -2.4, "Close"),
                              inputs1=filled.std_out, name="RepeatLoop", hide=True)
        jumping = BooleanMath(tree, location=(17.0, -2.0), operation="OR",
                              inputs0=skips.std_out, inputs1=repeats.std_out,
                              name="TakeJump", hide=True)

        # the destination is not searched for: it was worked out in python when
        # the graph was built and baked into a string with one character per
        # instruction, so reading it is the same slice-and-decode the
        # instruction itself goes through
        entry = SliceString(tree, location=(12.4, -3.4),
                            string=variables["Jumps"].std_out, position=counter,
                            length=1, name="JumpEntry")
        encoded = CharToAscii(tree, location=(13.6, -3.4), char=entry.std_out,
                              name="JumpCode")
        target = IntegerMath(tree, location=(14.8, -3.4), operation="SUBTRACT",
                             inputs0=encoded.std_out, inputs1=self.JUMP_ORIGIN,
                             name="JumpTarget")
        onward = IntegerMath(tree, location=(14.8, -4.2), operation="ADD",
                             inputs0=counter, inputs1=1, name="NextInstruction")
        jumped = Switch(tree, location=(18.0, -3.4), input_type="INT",
                        switch=jumping.std_out, false=onward.std_out,
                        true=target.std_out, name="CounterAfterStep")
        # on the frames in between two steps, and after the program has ended,
        # the counter stays where it is
        moved = Switch(tree, location=(19.0, -3.4), input_type="INT",
                       switch=fire, false=counter, true=jumped.std_out,
                       name="NewCounter")

        frame = Frame(tree, location=(12.0, 7.0), label="Automaton")
        frame.add(built + [stored, cell, raised, lowered, here, selection, tape,
                           place, letter, holds, prints, printed, output,
                           empty, filled, skips, repeats, jumping,
                           entry, encoded, target, onward, jumped, moved])
        return pointer.std_out, tape.geometry_out, output.std_out, moved.std_out

    # ----------------------------------------------------------------
    def _create_cells_frame(self, tree, control, run):
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
        quad = Quadrilateral(tree, location=(26, 2), mode="RECTANGLE",
                             width=control["CellSize"].std_out,
                             height=control["CellSize"].std_out)
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
                                material=control[node_name].std_out,
                                name="Paint" + node_name)
                    for column, ((node_name, _), selection)
                    in enumerate(zip(self.cell_colors, selections))]
        create_geometry_line(tree, [instances, realize] + painters)

        numbers = self._create_cell_values(tree, control, tape)
        joined = JoinGeometry(tree, location=(34, 2.6))
        tree.links.new(painters[-1].geometry_out, joined.geometry_in)
        tree.links.new(numbers, joined.geometry_in)
        # the tape lies in the x-y plane, which a camera looking along +y sees
        # edge-on. Laying it back brings the faces of the cells into view; the
        # numbers are pre-turned by the complement of this angle in
        # _create_cell_values, so that they come out upright.
        tilt = TransformGeometry(tree, location=(35, 2.6),
                                 rotation=[self.tape_tilt, 0, 0], name="LayTapeBack")
        create_geometry_line(tree, [joined, tilt])

        frame = Frame(tree, location=(25.6, 3.4), label="Cells")
        frame.add([quad, fill, instances, realize, value, here, holds, under,
                   joined, tilt] + painters)
        return tilt.geometry_out

    # ----------------------------------------------------------------
    def _create_cell_values(self, tree, control, tape):
        """``CellValues``: the number every cell holds, written on it.

        :return: the geometry socket of the numbers.
        """
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
                        inputs0=control["CellSize"].std_out, inputs1=self.glyph_size,
                        name="NumberSize")
        curves = StringToCurves(tree, location=(29, -1.4), string=digits.std_out,
                                size=size.std_out, align_x="CENTER", align_y="BOTTOM")
        realize = RealizeInstances(tree, location=(30, -1.4))
        fill = FillCurve(tree, location=(31, -1.4), mode="N-gons")
        painted = SetMaterial(tree, location=(32, -1.4),
                              material=control["GlyphColor"].std_out, name="PaintNumber")
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
        frame.add([value, position, zone, digits, size, curves, realize, fill,
                   painted, placed])
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
        table = control["CodeTable"]
        size = StringLength(tree, location=(-14.4, 16.6), string=table.std_out,
                            name="TableLength")
        zone = RepeatZone(tree, location=(-13, 16), node_width=8,
                          iterations=size.std_out)

        origin = SeparateXYZ(tree, location=(-12, 17.4),
                             vector=control["TablePosition"].std_out)
        column = MathNode(tree, location=(-12, 15.4), operation="MULTIPLY",
                          inputs0=zone.iteration, inputs1=self.table_spacing,
                          name="Column")
        across = MathNode(tree, location=(-11, 17.4), operation="ADD",
                          inputs0=origin.x, inputs1=column.std_out, name="AtColumn")
        # the number sits on the line of TablePosition, the letter one line below
        number_at = CombineXYZ(tree, location=(-9.5, 17.4), x=across.std_out,
                               y=origin.y, z=origin.z, name="NumberPosition")
        below = MathNode(tree, location=(-10.5, 16.2), operation="SUBTRACT",
                         inputs0=origin.z, inputs1=self.table_line_gap,
                         name="LetterLine")
        letter_at = CombineXYZ(tree, location=(-9.5, 16.2), x=across.std_out,
                               y=origin.y, z=below.std_out, name="LetterPosition")

        letter = SliceString(tree, location=(-12, 14.6), string=table.std_out,
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
        for curves, position, row, label in ((number_curves, number_at, 17.4, "Number"),
                                             (letter_curves, letter_at, 14.6, "Letter")):
            # String to Curves hands out instances of outlines; realizing and
            # filling them turns them into the solid letter that is drawn
            realize = RealizeInstances(tree, location=(-8, row))
            fill = FillCurve(tree, location=(-7, row), mode="N-gons")
            # the entry is one piece of geometry, not a field, so it can be
            # moved with Transform Geometry - Set Position would need it to be
            # an instance first and would then have to be realized again
            place = TransformGeometry(tree, location=(-6, row),
                                      translation=position.std_out,
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
                              material=control["GlyphColor"].std_out, name="PaintTable")
        create_geometry_line(tree, [joined, painted])

        frame = Frame(tree, location=(-14.6, 18.4), label="CodeTable")
        frame.add([size, zone, origin, column, across, number_at, below, letter_at,
                   letter, letter_curves, rank, number, number_curves, pair, grown,
                   joined, painted] + entries)
        return painted.geometry_out

    # ----------------------------------------------------------------
    def _create_table_frame(self, tree, control, table):
        """The rectangle around the code table, sized from what it contains.

        :return: the geometry socket of the rectangle.
        """
        bounds = BoundingBox(tree, location=(-3, 14.6), geometry=table)
        extent = VectorMath(tree, location=(-2, 15.2), operation="SUBTRACT",
                            inputs0=bounds.max_out, inputs1=bounds.min_out,
                            name="TableExtent")
        margin = VectorMath(tree, location=(-1, 15.2), operation="SCALE",
                            inputs0=extent.std_out, float_input=self.table_margin,
                            name="WithMargin")
        sides = SeparateXYZ(tree, location=(0, 15.2), vector=margin.std_out)
        middle = VectorMath(tree, location=(-2, 13.8), operation="ADD",
                            inputs0=bounds.min_out, inputs1=bounds.max_out,
                            name="TableCorners")
        centre = VectorMath(tree, location=(-1, 13.8), operation="SCALE",
                            inputs0=middle.std_out, float_input=0.5, name="TableCentre")
        # the table stands in the x-z plane, so its width and height are the x
        # and z of the bounding box, while the rectangle is born in x-y
        box = Quadrilateral(tree, location=(1, 14.6), mode="RECTANGLE",
                            width=sides.x, height=sides.z)
        # a bare curve renders as a hair thin enough to disappear, so the
        # rectangle is given a body before it is drawn
        wire = CurveWireFrame(tree, location=(2, 14.6), radius=self.frame_radius,
                              resolution=4, geometry=box.geometry_out)
        place = TransformGeometry(tree, location=(3, 14.6), translation=centre.std_out,
                                  rotation=[pi / 2, 0, 0], name="PlaceTableFrame")
        painted = SetMaterial(tree, location=(4, 14.6),
                              material=control["FrameColor"].std_out,
                              name="PaintTableFrame")
        create_geometry_line(tree, [wire, place, painted])

        frame = Frame(tree, location=(-4.4, 15.8), label="TableFrame")
        frame.add([bounds, extent, margin, sides, middle, centre, box,
                   wire, place, painted])
        return painted.geometry_out

    # ----------------------------------------------------------------
    def _create_display_frame(self, tree, control, label, width, position, location):
        """One of the three framed boxes below the tape.

        :param width: the ``Value`` node holding the width of the box
        :param position: the ``Vector`` node holding the middle of the box
        :param location: where the frame goes in the node editor
        :return: the geometry socket of the box.
        """
        x, y = location
        box = Quadrilateral(tree, location=(x, y), mode="RECTANGLE",
                            width=width.std_out, height=self.display_height)
        # a bare curve renders as a hair thin enough to disappear, so the
        # rectangle is given a body before it is drawn
        wire = CurveWireFrame(tree, location=(x + 1, y), radius=self.frame_radius,
                              resolution=4, geometry=box.geometry_out)
        place = TransformGeometry(tree, location=(x + 2, y), translation=position.std_out,
                                  rotation=[pi / 2, 0, 0], name="Place" + label)
        painted = SetMaterial(tree, location=(x + 3, y),
                              material=control["FrameColor"].std_out,
                              name="Paint" + label)
        create_geometry_line(tree, [place, painted], ins=wire.geometry_out)

        frame = Frame(tree, location=(x - 0.4, y + 0.8), label=label)
        frame.add([box, wire, place, painted])
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

        Each instruction is painted by what has become of it - see
        :attr:`PROGRAM_COLORS`. The one worth the trouble is ``WaitingColor``:
        an instruction that has run but sits inside a loop that is still open
        will run again, and :meth:`_loop_starts` says which those are. What is
        left in ``DoneColor`` is what has run for the last time, so the strip
        goes dark behind the box only where the machine is never coming back.

        The box marks the instruction *about to* run, not the one just run -
        the tape beside it is the state that instruction is about to act on,
        which is how a debugger shows the same thing.

        :return: the geometry socket of the strip and of the box on it.
        """
        program = variables["Input"].std_out
        counter = run["Counter"]
        size = StringLength(tree, location=(17, -21.4), string=program,
                            name="StripLength")

        # --- where the strip sits ---------------------------------------
        origin = SeparateXYZ(tree, location=(17, -22.2),
                             vector=control["InputPosition"].std_out)
        half = MathNode(tree, location=(18, -22.2), operation="MULTIPLY",
                        inputs0=control["InputDisplaySize"].std_out, inputs1=0.5,
                        name="HalfDisplay")
        edge = MathNode(tree, location=(19, -22.2), operation="SUBTRACT",
                        inputs0=origin.x, inputs1=half.std_out, name="DisplayLeftEdge")
        # A column per instruction, plus one at each end. The first is the
        # margin that keeps column 0 clear of the left edge; the second is
        # where the counter ends up when the program has run out, and the box
        # that marks it needs somewhere to park that is still inside the
        # display rather than astride its right edge.
        spacing = MathNode(tree, location=(18, -23), operation="DIVIDE",
                           inputs0=control["InputDisplaySize"].std_out,
                           inputs1=len(self.program) + 2, name="StripSpacing")
        first = MathNode(tree, location=(20, -22.2), operation="ADD",
                         inputs0=edge.std_out, inputs1=spacing.std_out,
                         name="FirstColumn")
        glyph = MathNode(tree, location=(19, -23), operation="MULTIPLY",
                         inputs0=spacing.std_out, inputs1=self.strip_glyph_size,
                         name="StripGlyphSize")

        # --- which loop is open where the counter stands -----------------
        entry = SliceString(tree, location=(17, -23.8), string=variables["Loops"].std_out,
                            position=counter, length=1, name="LoopEntry")
        encoded = CharToAscii(tree, location=(18, -23.8), char=entry.std_out,
                              name="LoopCode")
        opened = IntegerMath(tree, location=(19, -23.8), operation="SUBTRACT",
                             inputs0=encoded.std_out, inputs1=self.JUMP_ORIGIN,
                             name="OpenLoop")
        # 1-based, so 0 means the counter is not inside a loop at all and
        # nothing is waiting
        inside = CompareNode(tree, location=(20, -23.8), operation="GREATER_THAN",
                             data_type="INT", inputs0=opened.std_out, inputs1=0,
                             name="InsideALoop", hide=True)

        # --- one column per instruction ----------------------------------
        zone = RepeatZone(tree, location=(21, -21.4), node_width=9,
                          iterations=size.std_out)
        column = zone.iteration
        letter = SliceString(tree, location=(22, -22.2), string=program,
                             position=column, length=1, name="StripLetter")
        curves = StringToCurves(tree, location=(23, -22.2), string=letter.std_out,
                                size=glyph.std_out, align_x="CENTER",
                                align_y="MIDDLE", hide=True)
        realize = RealizeInstances(tree, location=(24, -22.2))
        fill = FillCurve(tree, location=(25, -22.2), mode="N-gons")
        # column n stands n spacings in from the first one, and stays there
        along = MathNode(tree, location=(23, -21.4), operation="MULTIPLY",
                         inputs0=column, inputs1=spacing.std_out,
                         name="StripOffset")
        across = MathNode(tree, location=(24, -21.4), operation="ADD",
                          inputs0=first.std_out, inputs1=along.std_out,
                          name="ColumnPosition")
        at = CombineXYZ(tree, location=(25, -21.4), x=across.std_out, y=origin.y,
                        z=origin.z, name="StripPlace")
        place = TransformGeometry(tree, location=(26, -22.2), translation=at.std_out,
                                  rotation=[pi / 2, 0, 0], name="PlaceColumn")

        done = CompareNode(tree, location=(22, -24.6), operation="LESS_THAN",
                           data_type="INT", inputs0=column, inputs1=counter,
                           name="HasRun", hide=True)
        # the "[" itself is not re-executed - "]" jumps back to the instruction
        # after it - so the block that is waiting starts one column further on
        after = CompareNode(tree, location=(22, -25.4), operation="GREATER_EQUAL",
                            data_type="INT", inputs0=column, inputs1=opened.std_out,
                            name="InTheBody", hide=True)
        within = BooleanMath(tree, location=(23, -25.4), operation="AND",
                             inputs0=inside.std_out, inputs1=after.std_out,
                             name="InAnOpenLoop", hide=True)
        waits = BooleanMath(tree, location=(24, -25.4), operation="AND",
                            inputs0=within.std_out, inputs1=done.std_out,
                            name="WillRunAgain", hide=True)
        now = CompareNode(tree, location=(22, -26.2), operation="EQUAL",
                          data_type="INT", inputs0=column, inputs1=counter,
                          name="IsCurrent", hide=True)

        selections = (None, done.std_out, waits.std_out)
        painters = [SetMaterial(tree, location=(27 + step, -22.2), selection=selection,
                                material=control[node_name].std_out,
                                name="Paint" + node_name)
                    for step, ((node_name, _), selection)
                    in enumerate(zip(self.program_colors, selections))]
        painters.append(SetMaterial(tree, location=(30, -22.2), selection=now.std_out,
                                    material=control["PointerColor"].std_out,
                                    name="PaintCurrentInstruction"))
        create_geometry_line(tree, [realize, fill, place] + painters,
                             ins=curves.geometry_out)

        grown = JoinGeometry(tree, location=(31, -22.2))
        tree.links.new(painters[-1].geometry_out, grown.geometry_in)
        tree.links.new(zone.repeat_input.outputs["Geometry"], grown.geometry_in)
        tree.links.new(grown.geometry_out, zone.repeat_output.inputs["Geometry"])

        frame = Frame(tree, location=(16.6, -20.6), label="ProgramStrip")
        frame.add([size, origin, half, edge, spacing, first, glyph, entry, encoded,
                   opened, inside, zone, letter, curves, realize, fill,
                   along, across, at, place, done, after, within, waits, now,
                   grown] + painters)

        cursor = self._create_cursor_frame(tree, control, counter, first, spacing,
                                           origin)
        both = JoinGeometry(tree, location=(33, -22.2))
        for piece in (zone.geometry_out, cursor):
            tree.links.new(piece, both.geometry_in)
        frame.add([both])
        return both.geometry_out

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
    def _create_simulated_geometry_frame(self, tree, control, variables, run):
        """``SimulatedGeometry``: everything that is redrawn every frame.

        What the machine has printed, written into its box, and the marker
        under the cell the head is on. This is built from the state the
        simulation zone *outputs*, not inside the zone: none of it is state, it
        is a picture of the state.

        :return: the geometry socket.
        """
        label, y = "OutputText", -8
        box_width, position = control["OutputDisplaySize"], control["OutputPosition"]
        # Plain text centred on the origin, *not* String to Curves' own
        # SCALE_TO_FIT: a text box hangs off the origin rather than surrounding
        # it, and where inside the box the text ends up moves with how far it
        # had to be shrunk - a long string comes out below the box that a two
        # letter one sits in the middle of. Centred text is in the same place
        # whatever it says, and the fitting is done below, where it can be
        # measured.
        curves = StringToCurves(tree, location=(26, y), string=run["Output"],
                                size=0.6 * self.display_height, align_x="CENTER",
                                align_y="MIDDLE", name=label, hide=True)
        realize = RealizeInstances(tree, location=(27, y))
        fill = FillCurve(tree, location=(28, y), mode="N-gons")
        # how much wider than its box the text came out
        bounds = BoundingBox(tree, location=(29, y - 1.4))
        extent = VectorMath(tree, location=(30, y - 1.4), operation="SUBTRACT",
                            inputs0=bounds.max_out, inputs1=bounds.min_out,
                            name="Extent" + label)
        across = SeparateXYZ(tree, location=(31, y - 1.4), vector=extent.std_out)
        # an empty string has no geometry and hence no width; the guard keeps
        # the division finite, and the MINIMUM below then leaves it alone at
        # scale 1
        wide = MathNode(tree, location=(32, y - 1.4), operation="MAXIMUM",
                        inputs0=across.x, inputs1=1e-3, name="Width" + label)
        ratio = MathNode(tree, location=(33, y - 1.4), operation="DIVIDE",
                         inputs0=box_width.std_out, inputs1=wide.std_out,
                         name="Ratio" + label)
        # only ever shrink: a short output should not be blown up to the full
        # width of its box
        factor = MathNode(tree, location=(34, y - 1.4), operation="MINIMUM",
                          inputs0=ratio.std_out, inputs1=1.0, name="Fit" + label)
        scale = CombineXYZ(tree, location=(35, y - 1.4), x=factor.std_out,
                           y=factor.std_out, z=factor.std_out, name="Scale" + label)
        place = TransformGeometry(tree, location=(29, y), translation=position.std_out,
                                  rotation=[pi / 2, 0, 0], scale=scale.std_out,
                                  name="Place" + label)
        create_geometry_line(tree, [realize, fill, place], ins=curves.geometry_out)
        tree.links.new(fill.geometry_out, bounds.geometry_in)
        pieces = [curves, realize, fill, bounds, extent, across, wide, ratio,
                  factor, scale, place]
        written = [place]

        # --- the head marker -------------------------------------------
        # the x of the cell is read off the tape rather than recomputed from
        # TapeSize and CellSize, so the marker cannot drift away from the cells
        # if the spacing of the Mesh Line is ever changed
        at = Position(tree, location=(26, -14))
        spot = SampleIndex(tree, location=(27, -14), data_type="FLOAT_VECTOR",
                           domain="POINT", geometry=run["Geometry"], value=at.std_out,
                           index=run["PointerPosition"], name="CellPosition")
        along = SeparateXYZ(tree, location=(28, -14), vector=spot.std_out)
        drop = SeparateXYZ(tree, location=(28, -14.8),
                           vector=control["PointerOffset"].std_out)
        under = CombineXYZ(tree, location=(29, -14), x=along.x, y=drop.y, z=drop.z,
                           name="MarkerPosition")
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
        put = TransformGeometry(tree, location=(29, -16), translation=under.std_out,
                                name="PlaceMarker")
        painted = SetMaterial(tree, location=(30, -16),
                              material=control["PointerColor"].std_out,
                              name="PaintMarker")
        create_geometry_line(tree, [marker, put, painted])

        # the three strings are painted together, and only then joined with the
        # marker: a Set Material without a selection paints everything it is
        # handed, so putting the marker in first would take its colour away
        lettering = JoinGeometry(tree, location=(33, -10))
        for piece in written:
            tree.links.new(piece.geometry_out, lettering.geometry_in)
        text = SetMaterial(tree, location=(34, -10),
                           material=control["GlyphColor"].std_out, name="PaintText")
        create_geometry_line(tree, [lettering, text])

        # the strip carries its own colours, one per instruction, so it joins
        # after the painting rather than before it
        strip = self._create_program_strip(tree, control, variables, run)
        joined = JoinGeometry(tree, location=(35, -12))
        for piece in (text.geometry_out, painted.geometry_out, strip):
            tree.links.new(piece, joined.geometry_in)

        frame = Frame(tree, location=(25.6, -7.2), label="SimulatedGeometry")
        frame.add(pieces + [at, spot, along, drop, under, tip, stem, lowered,
                            marker, put, painted, lettering, text, joined])
        return joined.geometry_out


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
        :attr:`CELL_COLORS` and :attr:`PROGRAM_COLORS`, and the two entries
        ``GlyphColor`` and ``FrameColor``
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

    # Colour of the instructions in the ascii table, and the colour everything
    # else in it is drawn in. The families are those of bff_trace.py, and match
    # ExtendedBrainFuckTapeModifier so that a tape drawn by that class and this
    # table can be shown together.

    OPCODE_COLORS = (
        # node name,        colour,             characters
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
            "Counter": InputInteger(tree, location=(x, -4.0), integer=0,
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
        inside = CompareNode(tree, location=(4.4, 3.6), operation="LESS_THAN",
                             data_type="INT", inputs0=sim_in.outputs["Counter"],
                             inputs1=cells, name="BeforeTheEnd", hide=True)
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

        numbers = self._create_cell_values(tree, control, tape)
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
        box = Quadrilateral(tree, location=(28, y - 1.2), mode="RECTANGLE",
                            width=side.std_out, height=side.std_out)
        # a bare curve renders as a hair thin enough to disappear
        wire = CurveWireFrame(tree, location=(29, y - 1.2),
                              radius=self.cursor_weight * self.cell_size,
                              resolution=6, geometry=box.geometry_out)
        place = TransformGeometry(tree, location=(30, y), translation=spot.std_out,
                                  rotation=[pi / 2 - self.tape_tilt, 0, 0],
                                  name="PlaceCursor")
        painted = SetMaterial(tree, location=(31, y),
                              material=control["PointerColor"].std_out,
                              name="PaintCursor")
        rides = StoredNamedAttribute(tree, location=(32, y), data_type="INT",
                                     domain="POINT", name="Tape", value=line.std_out,
                                     label="TapeOfCursor")
        create_geometry_line(tree, [place, painted, rides], ins=wire.geometry_out)
        pieces += [spot, line, side, box, wire, place, painted, rides]

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
    def _create_cell_values(self, tree, control, tape):
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
        value = NamedAttribute(tree, location=(26, -2), data_type="INT", name="Value")
        position = Position(tree, location=(25, -2.6))
        shift = VectorMath(tree, location=(26, -2.6), hide=True, operation="ADD", inputs0=position.std_out,
                           inputs1=Vector([0, 0, 0.25]))
        zone = ForEachZone(tree, location=(27, -1.4), domain="POINT", node_width=6,
                           geometry=tape)
        zone.add_socket(socket_type="INT", name="Value", value=value.std_out,
                        for_input=True)
        zone.add_socket(socket_type="VECTOR", name="Location", value=shift.std_out,
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

        glyph = Switch(tree, location=(29.4, -0.8), input_type="STRING",
                       switch=is_command, true=letter.std_out, false=digits.std_out,
                       name="CommandOrNumber")

        size = MathNode(tree, location=(28, -2.4), operation="MULTIPLY",
                        inputs0=control["CellSize"].std_out, inputs1=self.glyph_size,
                        name="NumberSize")
        bigger = MathNode(tree, location=(28.7, -2.4), operation="MULTIPLY",
                          inputs0=size.std_out, inputs1=self.cell_command_scale,
                          name="CommandSize")
        glyph_size = Switch(tree, location=(29.4, -2.4), input_type="FLOAT",
                            switch=is_command, true=bigger.std_out, false=size.std_out,
                            name="GlyphSize")

        curves = StringToCurves(tree, location=(30, -1.4), string=glyph.std_out,
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
        # which is what the chain of Set Material below selects on. Both lists
        # are built from opcode_colors in one comprehension so that they cannot
        # drift apart: two of its entries cover a *pair* of characters, so a
        # hand-written list of ten labels pairs up with the eight colours
        # wrongly and silently.
        #
        # The characters are in *single quotes*, and both of the jobs the
        # quotes do matter here. They keep "<" and ">" from being read as
        # LESS_THAN and GREATER_THAN, and they keep the "," from being read as
        # the separator between two tokens - see split_rpn in
        # geometry_nodes/nodes.py.
        #
        # "in" is Find in String's count of the entry inside the character set
        # of that colour - note the order, set first and letter second - so it
        # is 1 exactly when the entry is one of them, and a boolean socket
        # reads any non-zero as true. Writing it this way round is what lets
        # one formula serve both a single character and a pair.
        socket_labels = [node_name for node_name, _, _ in self.opcode_colors]
        color_selection = make_function(tree, name="ColorSelector", custom_ops=custom_ops,
                                        functions={
                                            node_name: "'%s',letter,in" % character
                                            for node_name, _, character
                                            in self.opcode_colors
                                        }, inputs=["letter"], outputs=socket_labels,
                                        strings=["letter"], booleans=socket_labels,
                                        hide=True, location=(-5, 14))
        tree.links.new(letter.std_out, color_selection.inputs["letter"])

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
        colour of an instruction (see ``BrainFuckExtendedModifier.OPCODE_COLORS``),
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
    OPCODE_COLORS = BrainFuckExtendedModifier.OPCODE_COLORS
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
        glyph_shift = VectorMath(tree,location=(-6,-9.6),label="GlyphShift",inputs0=position.std_out,inputs1=Vector([0,0,0.1]),hide=True)
        zone.add_socket(socket_type="VECTOR", name="Location",
                        value=glyph_shift.std_out, for_input=True)

        # the byte is this cell's character's ascii code (what the byte csv
        # stored); one Slice String into the ascii table turns it back into the
        # character, and everything downstream is exactly what the string
        # version fed - a Find in String colour test and String to Curves.
        letter = SliceString(tree, location=(-3, -8), string=control["AsciiTable"].std_out,
                             position=zone.foreach_input.outputs["Byte"],
                             length=1, name="Letter")

        # "in" is Find in String's count of the letter inside a character
        # set - see BrainFuckExtendedModifier._create_code_table_frame for
        # the same pattern applied to the ascii table instead of a tape
        custom_ops = {
            "in": {"type": FindInString, "inputs": ("String", "Search"),
                   "output": "Count", "label": "in"},
        }
        socket_labels = [node_name for node_name, _, _ in self.opcode_colors]
        color_selection = make_function(
            tree, name="ColorSelector", custom_ops=custom_ops,
            functions=dict(
                {node_name: "'%s',letter,in" % character
                 for node_name, _, character in self.opcode_colors},
                IsOperator="commands,letter,in,0,>"),
            inputs=["letter", "commands"], outputs=socket_labels + ["IsOperator"],
            strings=["letter", "commands"], booleans=socket_labels + ["IsOperator"],
            hide=True, location=(-3, -9.4))
        tree.links.new(letter.std_out, color_selection.inputs["letter"])
        tree.links.new(control["Operators"].std_out, color_selection.inputs["commands"])
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

        placed_glyph = TransformGeometry(tree, location=(4, -8),rotation=Vector([pi/2,0,0]),
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

TRACK_X_IN = 40.0  # the track starts here, well off screen to the right
TRACK_Z_UP = 1.6  # height of the incoming run, in the upper half
TRACK_X_LOOP = 1.0  # where the roller-coaster loop sits
TRACK_R1 = 2.7  # and how big it is
TRACK_DEPTH = 4.5  # how far back in y a loop steps to clear its own entry
TRACK_X_LEFT = -11.0  # how far left it gets before turning back
TRACK_R2 = 2.8  # radius of the 180 degree turn that sends it back right
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
TRACK_Z_OUT = -1.0
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

    def straight(x0, z0, x1, z1, y0, y1, n, y_over=1.0):
        """One straight run, with ``y`` and ``z`` eased across it.

        :param y_over: the fraction of the segment ``y`` makes its move in.
            The default spends the whole segment on it; a smaller value gets
            the move done early and then holds, which is how the last straight
            is over the gate while it is still on screen rather than half a
            frame width off the right edge.
        """
        for k in range(1, n + 1):
            u = k / n
            pts.append((x0 + (x1 - x0) * u,
                        y0 + (y1 - y0) * _smoothstep(u / y_over),
                        z0 + (z1 - z0) * _smoothstep(u)))

    tau = 2.0 * pi

    # 1 - fly in from the right
    straight(TRACK_X_IN, TRACK_Z_UP, TRACK_X_LOOP, TRACK_Z_UP, 0.0, 0.0, 400)
    ends.append(("loop_start", len(pts) - 1))

    # 2 - the roller-coaster loop
    n = 400
    for k in range(1, n + 1):
        a = tau * k / n
        pts.append((TRACK_X_LOOP - TRACK_R1 * math.sin(a),
                    TRACK_DEPTH * _smoothstep(a / tau),
                    TRACK_Z_UP + TRACK_R1 - TRACK_R1 * math.cos(a)))
    ends.append(("loop_end", len(pts) - 1))

    # 3 - on to the left, climbing to the height the turn needs to start at,
    # and back out of the loop's depth
    straight(TRACK_X_LOOP, TRACK_Z_UP, TRACK_X_LEFT, TRACK_Z_MID,
             TRACK_DEPTH, 0.0, 400)
    ends.append(("turn_start", len(pts) - 1))

    # 4 - the 180 degree turn, ending level and heading right. It also spends
    # itself getting y up to the near edge of the gate, so that the molecule
    # comes out of the turn with the fork immediately ahead of it.
    n = 600
    for k in range(1, n + 1):
        b = 0.5 * tau * k / n
        pts.append((TRACK_X_LEFT - TRACK_R2 * math.sin(b),
                    TRACK_Y_WOUND * _smoothstep(b / (0.5 * tau)),
                    TRACK_Z_MID - TRACK_R2 + TRACK_R2 * math.cos(b)))
    ends.append(("turn_end", len(pts) - 1))

    # 5 - away to the right and off the screen, crossing the gate as it goes
    # and staying over it: anything that drifted back under TRACK_Y_WOUND here
    # would wind itself up again on the way out.
    z_out = TRACK_Z_OUT
    straight(TRACK_X_LEFT, z_out, TRACK_X_OUT, z_out,
             TRACK_Y_WOUND, TRACK_Y_OPEN, 900,
             y_over=TRACK_OPEN_RUN / (TRACK_X_OUT - TRACK_X_LEFT))

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
    return even, total, {name: cumulative[i] for name, i in ends}


def write_dna_track(path):
    """Put the track where an ``Import CSV`` node can read it.

    The first line is spent on the column header - ``Import CSV`` always does
    that and has no option not to - so the columns arrive in the graph as three
    float attributes called X, Y and Z.

    :return: ``(length, marks)`` - see :func:`dna_flight_path`.
    """
    points, total, marks = dna_flight_path()
    with open(path, "w") as file:
        file.write("X,Y,Z\n")
        for x, y, z in points:
            file.write("%.5f,%.5f,%.5f\n" % (x, y, z))
    return total, marks


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

    Two things follow from doing it by index rather than by any kind of
    matching, and both are the node tree's own doing rather than choices
    made here:

    - the point that ends up at the arrow's tip is whichever point of the
      frame happens to share the tip's index, so the frame turns itself
      inside out on the way rather than folding neatly;
    - the two shapes do not have the same number of points (the frame's tube
      has more), and an index past the end of the arrow does not clamp - it
      samples as the zero vector - so the frame's surplus points all collapse
      onto the world origin. The morph therefore arrives at an arrow *plus* a
      knot of geometry at ``(0, 0, 0)``, which is right at the foot of the
      arrow and so easy to mistake for part of it. Give the profile circle
      fewer segments, or the cone and cylinder more, if the counts should
      meet - see :meth:`point_counts`, which reports both.

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
                 arrow_resolution=32, morph=0.0, color=None,
                 name="Morph", **kwargs):
        self.color = color
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
                          name="Morph")

        node_frame = Frame(tree, location=(4.5, -0.5), label="Morphing")
        node_frame.add([parameter, morph])
        return morph.geometry_out


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

    BASE_COLORS = ("custom1", "joker", "important", "drawing")

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
        self.track_length, self.track_marks = write_dna_track(self.track_path)
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
    BASE_COLORS = DNAModifier.BASE_COLORS[:3] + ("example",)

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
    BASE_COLORS = DNAModifier.BASE_COLORS[:3] + ("example",)

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
        tangent = InputTangent(tree, location=at(5.7, -0.1), node_height=GRID,
                               name="Tangent", label="Tangent")
        plane = InputVector(tree, location=at(5.4, -0.4),
                            vector=Vector([0.0, 1.0, 0.0]), node_height=GRID,
                            name="PlaneNormal", label="PlaneNormal")
        spoke = VectorMath(tree, location=at(6.6, -0.8),
                           operation="CROSS_PRODUCT", inputs0=tangent.std_out,
                           inputs1=plane.std_out, node_height=GRID,
                           name="InwardSpoke", label="InwardSpoke")
        normal = VectorMath(tree, location=at(7.4, -1.1), operation="SCALE",
                            inputs0=spoke.std_out,
                            float_input=control["Radius"].std_out,
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
                   tangent, plane, spoke, normal, index, capture])
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
