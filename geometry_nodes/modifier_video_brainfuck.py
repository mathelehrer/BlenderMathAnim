import math
import os

from geometry_nodes.geometry_nodes_modifier import GeometryNodesModifier

from geometry_nodes.nodes import Points, InputValue, InstanceOnPoints, JoinGeometry, \
    create_geometry_line, RealizeInstances, Position, make_function, Index, SetMaterial, \
    RepeatZone, StoredNamedAttribute, NamedAttribute, VectorMath, TransformGeometry, InputVector, MeshLine, BooleanMath, \
    Simulation, MathNode, CombineXYZ, Switch, CylinderMesh, ConeMesh, Frame, SeparateXYZ, ForEachZone, Quadrilateral, \
    InputInteger, \
    CurveWireFrame, ValueToString, StringToCurves, FillCurve, CompareNode, \
    InputMaterial, BoundingBox, InputString, SampleIndex, StringJoin, SliceString, Reroute, CharToAscii, \
    StringLength, IntegerMath, FindInString, SetPosition, ImportCSV, NodeGroup, \
    DomainSize, CombineBundle, SeparateBundle, GetBundleItem, SceneTime, ExtrudeMesh
from interface.ibpy import Vector
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
        rest =[("GlyphColor", self.glyph_color),
                                            ("FrameColor", self.frame_color)]
        for offset, (node_name, color) in enumerate(rest):
            palette[node_name] = InputMaterial(
                tree, location=(x, -4.4 - 0.4 * (len(self.cell_colors) + offset)),
                material=color, name=node_name,hide=True, **self.kwargs)
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
        self.tape_columns = tuple(csv_column(os.path.join(DATA_DIR, file_name+".csv"))
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
        for piece in [cells, table]:#, simulated]:
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
                                           path=os.path.join(DATA_DIR, file_name+".csv"),
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

            length = MathNode(tree, location=(-8, -3*i), operation="MULTIPLY",
                              inputs0=control["TapeSize"].std_out,
                              inputs1=control["CellSize"].std_out, name="TapeLength")
            end = CombineXYZ(tree, location=(-7, -3*i), x=length.std_out, name="TapeEnd")
            line = MeshLine(tree, location=(-6, 0.6-3*i), mode="END_POINTS",
                            count=control["TapeSize"].std_out,
                            start_location=Vector([0, 0, 0]), end_location=end.std_out)
            # what the file holds for this cell. The column of the point cloud
            # is named after the header line of the csv file, and the index is
            # the index of the cell being written - a cell beyond the end of
            # the file gets a zero, so a short file simply leaves the rest of
            # the tape blank.
            column = NamedAttribute(tree, location=(-6, -0.6-3*i), data_type="INT",
                                    name=self.tape_columns[i], label="CsvColumn")
            cell = Index(tree, location=(-6, -1.2-3*i), name="CellIndex", hide=True)
            content = SampleIndex(tree, location=(-5.4, 0.6-3*i), data_type="INT",
                                  domain="POINT", geometry=control[self.TAPE_SOURCES[i]].geometry_out,
                                  value=column.std_out, index=cell.std_out,
                                  label="ReadCell" + str(i))
            # the attribute has to exist from the first frame on, otherwise the
            # "Sample Index" in the automaton has nothing to read and the cells
            # have nothing to be coloured by.
            values = StoredNamedAttribute(tree, location=(-4.6, 0.6-3*i), data_type="INT",
                                          domain="POINT", name="Value", value=content.std_out,
                                          label="LoadTape")

            tape_kind = StoredNamedAttribute(tree, location=(-3.6, 0.6-3*i), data_type="INT",
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
                IntegerMath(tree, location=(-3.4, -0.4-3*i), operation="ADD",
                            inputs0=cell.std_out, inputs1=control["TapeSize"].std_out,
                            name="CellOnTape" + str(i), hide=True)]
            number = StoredNamedAttribute(tree, location=(-3.0, 0.6-3*i), data_type="INT",
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

        tape_shift = make_function(tree,name="TapeShift",
                    functions={
                        "translation":["cell_width,cell_number,*,-2,/","0","0"],
                        "offset":["0","0","-4,tape,*"]
                    },inputs=["cell_width","cell_number","tape"],outputs=["translation","offset"],
                    scalars=["cell_width","cell_number"],integers=["tape"],vectors=["translation","offset"],
                                   hide=True,location=(34,1.6))
        tree.links.new(control["CellSize"].std_out,tape_shift.inputs["cell_width"])
        tree.links.new(control["TapeSize"].std_out,tape_shift.inputs["cell_number"])
        tree.links.new(attr_tape.std_out,tape_shift.inputs["tape"])

        tilt = TransformGeometry(tree, location=(35, 2.6),translation=tape_shift.outputs["translation"],
                                 rotation=[self.tape_tilt, 0, 0], name="LayTapeBack")

        set_position = SetPosition(tree,location=(36,2),offset=tape_shift.outputs["offset"])

        create_geometry_line(tree, [joined, tilt,set_position])

        frame = Frame(tree, location=(25.6, 3.4), label="Cells")
        frame.add([quad, fill, instances, realize, value, here, holds, under,
                   joined, tilt,set_position,attr_tape,tape_shift] + painters)
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
    the layout math, ``FindInString`` to test a character against an
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
      which snapshot's 6400 characters of the data file are on screen.

    A cell shows nothing unless its byte was one of the ten BFF instructions
    when ``soup_watcher.py`` recorded it: that is what ``soup.render()``
    already encodes (an instruction's own character, ``'0'`` for a zero
    byte, ``' '`` for anything else), so a blank cell here is a cell
    ``Switch`` never lets through to ``String to Curves`` at all, not a cell
    drawn and then hidden.

    :param data_file: name of the file ``soup_watcher.py`` wrote, resolved
        against ``DATA_DIR`` unless it is already an absolute path
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
    TAPE_COLOR = "gray_4"

    def __init__(self, data_file="soup_evolution.csv", max_snapshots=None,
                cell_size=0.09, column_gap=1.2, row_spacing=0.13,
                tape_tilt=0.4607669, glyph_size=0.85, stick_out=0.05,
                frames_per_snapshot=10, colors=None, name="SoupWatcher",
                **kwargs):
        self.cell_size = cell_size
        self.column_gap = column_gap
        self.row_spacing = row_spacing
        self.tape_tilt = tape_tilt
        self.glyph_size = glyph_size
        self.stick_out = stick_out
        self.frames_per_snapshot = frames_per_snapshot

        overrides = colors or {}
        self.opcode_colors = tuple((node_name, overrides.get(node_name, color), character)
                                   for node_name, color, character in self.OPCODE_COLORS)
        self.glyph_color = overrides.get("GlyphColor", self.GLYPH_COLOR)
        self.tape_color = overrides.get("TapeColor", self.TAPE_COLOR)

        path = data_file if os.path.isabs(data_file) else os.path.join(DATA_DIR, data_file)
        self.tapes, self.num_snapshots = self._load_snapshots(path, max_snapshots)

        self.kwargs = kwargs
        super().__init__(name=name, automatic_layout=False)

    # ----------------------------------------------------------------
    @classmethod
    def _load_snapshots(cls, path, max_snapshots):
        """Read ``path`` and concatenate whole snapshots of 100 tapes.

        :return: ``(all_tapes, num_snapshots)`` - one string, every
            snapshot's 100 tapes of 64 characters each, back to back in the
            order ``soup_watcher.py`` appended them, and how many of them
            there are.
        """
        with open(path) as file:
            lines = [line.rstrip("\n") for line in file if line.rstrip("\n")]
        for i, line in enumerate(lines):
            if len(line) != cls.TAPE:
                raise ValueError(
                    "%s line %d: expected %d characters (one tape), got %d"
                    % (path, i, cls.TAPE, len(line)))
        num_snapshots = len(lines) // cls.TAPES_PER_SNAPSHOT
        if max_snapshots is not None:
            num_snapshots = min(num_snapshots, max_snapshots)
        if num_snapshots < 1:
            raise ValueError("%s has fewer than %d tapes (one snapshot)"
                             % (path, cls.TAPES_PER_SNAPSHOT))
        lines = lines[:num_snapshots * cls.TAPES_PER_SNAPSHOT]
        return "".join(lines), num_snapshots

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
            # the whole data file, every snapshot's tapes back to back - see
            # :meth:`_load_snapshots`. A cell's character is read out of this
            # with Slice String rather than one Import CSV row per cell: the
            # existing tape_files convention is one *number* per row, and
            # this data is a fixed-width block of *text* instead.
            "Tapes": InputString(tree, location=(x, -1.8), string=self.tapes,
                                 name="Tapes", hide=True),
            "Operators": InputString(tree, location=(x, -2.4), string=self.OPERATORS,
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
        """How far into ``control["Tapes"]`` the block on screen right now starts.

        A pure function of the current frame - no simulation zone needed,
        since nothing here accumulates: the block index is
        ``floor(frame / FramesPerSnapshot) mod NumSnapshots``, and the offset
        is that many whole snapshots' worth of characters.

        :return: an INT socket, the character offset of the current snapshot.
        """
        frame_now = SceneTime(tree, location=(-8, 3), std_out="Frame", hide=True)
        offset = make_function(
            tree, name="SnapshotOffset",
            functions={
                "offset": "frame,fps,/,floor,n,%%,%d,*" % (self.TAPES_PER_SNAPSHOT * self.TAPE)
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
        """``TapeBars``: a flat background strip for each of the 100 tapes.

        One point per tape - ``Index`` is the tape number, ``0`` to ``99`` -
        positioned by column and row exactly as :meth:`_create_glyphs_frame`
        positions a tape's cells, so a glyph and the strip under it always
        agree on where their tape sits. A single rectangle is instanced onto
        all hundred: the strips carry no information of their own, unlike
        the glyphs, so they need only one shared colour between them.

        :return: the geometry socket of the hundred strips.
        """
        n = self.TAPES_PER_SNAPSHOT
        index = Index(tree, location=(-8, -2), hide=True)
        layout = make_function(
            tree, name="TapeBarPosition",
            aux_functions={
                "row": "index,%d,%%" % self.ROWS,
                "col": "index,%d,/,floor" % self.ROWS,
            },
            functions={
                # the strip is centred on its point, so it needs to sit half
                # a tape's width past the left edge of its column
                "position": [
                    "col,%d,cellSize,*,columnGap,+,*,%d,cellSize,*,2,/,+" % (self.TAPE, self.TAPE),
                    "0",
                    "row,rowSpacing,*,-1,*",
                ],
            },
            inputs=["index", "cellSize", "columnGap", "rowSpacing"],
            outputs=["position"], vectors=["position"],
            scalars=["index", "cellSize", "columnGap", "rowSpacing", "row", "col"],
            hide=True, location=(-7, -2))
        tree.links.new(index.std_out, layout.inputs["index"])
        tree.links.new(control["CellSize"].std_out, layout.inputs["cellSize"])
        layout.inputs["columnGap"].default_value = self.column_gap
        layout.inputs["rowSpacing"].default_value = self.row_spacing

        points = Points(tree, location=(-6, -2), count=n)
        placed = SetPosition(tree, location=(-5, -2), position=layout.outputs["position"])
        create_geometry_line(tree, [points, placed])

        width = MathNode(tree, location=(-6, -3), operation="MULTIPLY",
                         inputs0=control["CellSize"].std_out, inputs1=0.99 * self.TAPE,
                         name="BarWidth", hide=True)
        height = MathNode(tree, location=(-6, -3.4), operation="MULTIPLY",
                          inputs0=control["CellSize"].std_out, inputs1=0.7,
                          name="BarHeight", hide=True)
        bar = Quadrilateral(tree, location=(-5, -3), mode="RECTANGLE",
                            width=width.std_out, height=height.std_out)
        fill = FillCurve(tree, location=(-4, -3), mode="N-gons")
        create_geometry_line(tree, [bar, fill])

        instances = InstanceOnPoints(tree, location=(-3, -2), points=placed.geometry_out,
                                     instance=fill.geometry_out)
        realize = RealizeInstances(tree, location=(-2, -2))
        painted = SetMaterial(tree, location=(-1, -2), material=control["TapeColor"].std_out,
                              name="PaintTapes")
        create_geometry_line(tree, [instances, realize, painted])

        frame = Frame(tree, location=(-8.2, -1.4), label="TapeBars")
        frame.add([index, layout, points, placed, width, height, bar, fill, instances,
                  realize, painted])
        return painted.geometry_out

    # ----------------------------------------------------------------
    def _create_glyphs_frame(self, tree, control, snapshot_offset):
        """``Glyphs``: the operator that shows through each cell, if any.

        One point per cell of every tape in the snapshot -
        :attr:`TAPES_PER_SNAPSHOT` times :attr:`TAPE` of them, laid out the
        same way :meth:`_create_tape_bars_frame` lays out the tapes
        themselves - then a ``ForEachZone`` reads the one character
        ``snapshot_offset`` and this cell's own flattened index select out of
        ``control["Tapes"]``, keeps it only if it is one of the ten
        instructions (blank otherwise), and turns what is left into a letter
        standing extruded out of the tape rather than lying flat on it.

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
                    "0",
                    "row,rowSpacing,*,-1,*",
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

        global_index = IntegerMath(tree, location=(-6, -9), operation="ADD",
                                   inputs0=snapshot_offset, inputs1=index.std_out,
                                   name="GlobalIndex", hide=True)
        position = Position(tree, location=(-6, -9.6), hide=True)

        zone = ForEachZone(tree, location=(-4, -8), domain="POINT", node_width=9,
                           geometry=placed.geometry_out)
        zone.add_socket(socket_type="INT", name="GlobalIndex",
                        value=global_index.std_out, for_input=True)
        zone.add_socket(socket_type="VECTOR", name="Location",
                        value=position.std_out, for_input=True)

        letter = SliceString(tree, location=(-3, -8), string=control["Tapes"].std_out,
                             position=zone.foreach_input.outputs["GlobalIndex"],
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

        placed_glyph = TransformGeometry(tree, location=(4, -8),
                                         translation=zone.foreach_input.outputs["Location"],
                                         name="PlaceGlyph")
        zone.create_geometry_line([realize, fill, stick_out] + painters + [placed_glyph],
                                  ins=curves.geometry_out)

        frame = Frame(tree, location=(-8.2, -7.4), label="Glyphs")
        frame.add([index, layout, points, placed, global_index, position, zone, letter,
                  color_selection, glyph, size, curves, realize, fill, stick_out,
                  opcolors, placed_glyph] + painters)
        return zone.geometry_out

