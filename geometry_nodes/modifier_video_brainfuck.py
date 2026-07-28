import math

from geometry_nodes.geometry_nodes_modifier import GeometryNodesModifier

from geometry_nodes.nodes import Points, InputValue, InstanceOnPoints, JoinGeometry, \
    create_geometry_line, RealizeInstances, Position, make_function, Index, SetMaterial, \
    RepeatZone, StoredNamedAttribute, NamedAttribute, VectorMath, TransformGeometry, InputVector, MeshLine, BooleanMath, \
    Simulation, MathNode, CombineXYZ, Switch, CylinderMesh, ConeMesh, Frame, SeparateXYZ, ForEachZone, Quadrilateral, InputInteger, \
    CurveWireFrame, ValueToString, StringToCurves, FillCurve, CompareNode, \
    InputMaterial, BoundingBox, InputString, SampleIndex, StringJoin, SliceString, Reroute, CharToAscii, \
    StringLength, IntegerMath, FindInString
from interface.ibpy import Vector

pi = math.pi


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
    PROGRAM_COLORS = (
        ("ProgramColor", "text"),  # still to come
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
        rest = list(self.program_colors) + [("GlyphColor", self.glyph_color),
                                            ("FrameColor", self.frame_color)]
        for offset, (node_name, color) in enumerate(rest):
            palette[node_name] = InputMaterial(
                tree, location=(x, -4.4 - 0.4 * (len(self.cell_colors) + offset)),
                material=color, name=node_name, **self.kwargs)
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


class BrainFuckExtendedModifier(GeometryNodesModifier):
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

    ALPHABET = " !" + chr(
        34) + r"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"

    COMMANDS = ".,<>[]{}+-"
    # how much bigger an instruction is drawn than the rest of the alphabet
    command_glyph_scale = 4
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
        ("LessColor", "drawing", "<"),
        ("MoreColor", "cyan", ">"),
        ("CurlyBraceOpenColor", "some_logo_blue", "{"),
        ("CurlyBraceClosedColor", "some_logo_blue", "}"),
        ("PlusColor", "important", "+"),
        ("MinusColor", "orange", "-"),
        ("DotColor", "joker", "."),
        ("CommaColor", "some_logo_green", ","),
        ("BracketOpenColor", "x14_color", "["),
        ("BracketClosedColor", "x14_color", "]"),
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

    def __init__(self, program=None, code_table=None, table_width=30, tape_size=5, cell_size=1,
                 step_duration=0.5, start_time=3.0, tape_tilt=0.4607669,
                 glyph_size=0.6, display_height=2.0, colors=None,
                 name="SimpleBrainFuck", **kwargs):
        self.program = self.HELLO if program is None else program
        self.jumps = self._encode_jumps(self.program)
        self.loops = self._encode_loop_starts(self.program)
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
        # Coord(tree, min=(-10, 0), max=(10, 20))

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
            "TableWidth": InputInteger(tree, location=(x, -0.8), integer=self.table_width,hide=True),
            "TapeSize": InputInteger(tree, location=(x, 0), integer=self.tape_size,
                                     name="TapeSize",hide=True),
            "CellSize": InputValue(tree, location=(x, -0.8), value=self.cell_size,
                                   name="CellSize",hide=True),
            "StartTime": InputValue(tree, location=(x, -1.6), value=self.start_time,
                                    name="StartTime",hide=True),
            "StepDuration": InputValue(tree, location=(x, -2.4), value=self.step_duration,
                                       name="StepDuration",hide=True),
            "CodeTable": InputString(tree, location=(x, -3.2), string=self.code_table,
                                     name="CodeTable",hide=True),
            "CommandTable": InputString(tree, location=(x, -3.8), string=self.command_table, name="CommandTable",hide=True)
        }

        # one Input Material node per colour of a cell, plus the two that
        # everything else is drawn in
        palette = {}
        rows = ([(node_name, color) for node_name, color in self.cell_colors]
                + [(node_name, color) for node_name, color, _ in self.opcode_colors])

        for row, (node_name, color) in enumerate(rows):
            palette[node_name] = InputMaterial(tree, location=(x, -4.4 - 0.4 * row),
                                               material=color, name=node_name,
                                               **self.kwargs,hide=True)
        rest = list(self.program_colors) + [("GlyphColor", self.glyph_color),
                                            ("FrameColor", self.frame_color)]
        for offset, (node_name, color) in enumerate(rest):
            palette[node_name] = InputMaterial(
                tree, location=(x, -4.4 - 0.4 * (len(rows) + offset)),
                material=color, name=node_name, **self.kwargs)
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
        table_start = - 0.5 * (self.table_width - 1) * self.table_spacing
        control["TablePosition"] = InputVector(tree, location=(x, -11.0),
                                               vector=Vector([table_start, 0, 10]),
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

        painters =([SetMaterial(tree,location=[-2,16.5],material=control["ZeroColor"].std_out,hide=True,name="PaintDefault")]+
                   [SetMaterial(tree, location=(-2, 16 - 0.5 * row), selection=selection,
                                material=control[node_name].std_out,
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
                   pair, grown, joined] + entries)
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


class BrainFuckSimpleExtended(BrainFuckSimpleModifier):
    """
    The two-headed machine of the BFF paper: a tape of two rows of 64 cells
    with an arrow under it and an arrow over it, and the printable ascii table
    drawn above as the legend for what the numbers in the cells mean.

    What it adds to :class:`SimpleBrainFuckModifier`:

    ``{`` and ``}``
        move the *upper* arrow, exactly as ``<`` and ``>`` move the lower one.
        Nothing else touches it - it is a second head with no arithmetic of its
        own.

    ``.`` and ``,``
        no longer print. ``.`` copies the value of the cell under the lower
        arrow into the cell under the upper one, and ``,`` copies it back the
        other way. That is the whole of the machine's data movement, and it is
        what makes a program able to write another program - the point of the
        exercise in ``brainfuck/bff/``.

    ``+``, ``-``, ``<``, ``>``, ``[``, ``]``
        unchanged, and all of them work on the lower arrow. In particular a
        loop tests the cell under the *lower* arrow.

    **The tape.** ``columns * rows`` cells, one run of memory laid out over two
    lines because 128 cells in a row would be unreadable: cell 0 to 63 on the
    upper line, 64 to 127 on the lower one, which is how the two 64-byte
    programs of the paper sit in the 128-byte tape. Both arrows range over the
    whole of it, so either can be on either line, and the gaps between the
    lines and between the tape and the ascii table are wide enough to hold an
    arrow that has nowhere else to be - including the case where the lower
    arrow points up at the upper line while the upper arrow points down at the
    lower line, in the same column, when the two are nose to nose in the same
    gap.

    **The ascii table.** The 95 printable characters in code order. The ten
    that mean something to the machine are drawn thickened and in the colour
    their family has in ``video_bff/bff_trace.py`` - blues move a head, reds do
    arithmetic, greens copy, violet is control flow - and the rest are left
    thin and grey. A code is written above every tenth entry rather than above
    all of them: 95 three-digit numbers across the width of the tape come out
    at a few pixels each, and the ten that are there are enough to count from.

    Thickening is done by drawing the outline of the glyph as well as filling
    it, which is a genuine bolder weight rather than a bigger one and needs no
    second font.

    **What is not here.** There is no input display and no output display. The
    program is not drawn - see :class:`SimpleBrainFuckModifier` for the strip
    that does that - and there is nothing to print, so the box that used to
    hold the output is the second arrow instead.

    The tape is drawn flat and face-on rather than laid back like the one-line
    machine: two lines seen at an angle would sit at two different distances
    from the camera, and the point of two lines is that they read as one tape
    folded once.

    :param program: the program, in ``> < { } + - . , [ ]``
    :param columns: cells per line
    :param rows: lines
    :param cell_size: width and height of a single cell
    :param head: where the lower arrow starts
    :param mate: where the upper arrow starts; defaults to the first cell of
        the second line, so that the two are visibly apart
    :param step_duration: seconds one instruction is on screen
    :param start_time: seconds before the first instruction runs
    :param glyph_size: height of the number on a cell, as a fraction of
        ``CellSize``
    :param row_gap: distance between the two lines, in cells
    :param table_gap: distance from the upper line to the ascii table, in cells
    :param colors: optional ``{node name: colour name}`` overriding
        :attr:`CELL_COLORS` and :attr:`OPCODE_COLORS`
    """

    # Writes 66, 70, 70 - "B", "F", "F" in the ascii table - into the first
    # three cells of the second line, and then copies the first of them back
    # into the cell the lower arrow is on. Every one of the ten instructions is
    # used: the 65 is built by a loop rather than by 65 "+", the three values
    # are moved across with ".", the upper arrow is walked along with "}" and
    # back with "{", and the last "," brings a value back the other way.
    WRITE_BFF = "+++++++++++++[>+++++<-]>+.}++++.}.{{,"

    # Colour of a cell by what is in it and which arrow is on it. Applied in
    # this order, each overriding the last.
    CELL_COLORS = (
        ("ZeroColor", "gray_1"),  # nothing in it yet
        ("ValueColor", "drawing"),  # holds a value
        ("LowerColor", "important"),  # the cell the lower arrow is on
        ("UpperColor", "joker"),  # the cell the upper arrow is on
    )
    # Colour of the instructions in the ascii table, and the colour everything
    # else in it is drawn in. The families are those of bff_trace.py, and match
    # ExtendedBrainFuckTapeModifier so that a tape drawn by that class and this
    # table can be shown together.
    OPCODE_COLORS = (
        # node name,        colour,             characters
        ("LessColor", "drawing", "<"),
        ("MoreColor", "cyan", ">"),
        ("CurlyBraceColor", "some_logo_blue", "{}"),
        ("PlusColor", "important", "+"),
        ("MinusColor", "orange", "-"),
        ("DotColor", "joker", "."),
        ("CommaColor", "some_logo_green", ","),
        ("BracketColor", "x14_color", "[]"),
    )
    GLYPH_COLOR = "text"  # the numbers in the cells
    ALPHABET_COLOR = "gray_4"  # the ascii characters that mean nothing
    FRAME_COLOR = "gray_2"  # the box around the ascii table

    # ascii codes of the two instructions the one-headed machine does not have
    BRACE_LEFT, BRACE_RIGHT = ord("{"), ord("}")
    COMMA = ord(",")

    # the printable range, which is the whole of the table
    FIRST_PRINTABLE = CharToAscii.FIRST_PRINTABLE
    LAST_PRINTABLE = CharToAscii.LAST_PRINTABLE

    # how the ascii table is drawn: the characters spread over the width of the
    # tape, a code above every ``table_label_every``-th of them, and a frame
    # around the lot
    table_glyph_size = 0.62
    table_line_gap = 1.0
    table_label_size = 0.5
    table_label_every = 10
    table_margin = 1.06
    bold_weight = 0.022  # outline radius, as a fraction of the glyph size
    frame_radius = 0.03

    # how much of its cell the drawn square fills. The cells sit one cell_size
    # apart, so a square of the full size would touch its neighbours and 64 of
    # them would read as one bar rather than as cells.
    cell_fill = 0.86

    # The arrows, in cells. Both gaps below are set so that these fit with
    # room to spare - see row_gap and table_gap. An arrow is built about its
    # middle: blender's cone runs from z=0 to z=depth while its cylinder is
    # centred on the origin, so the stem is dropped by half its own length to
    # meet the base of the head rather than by half the arrow.
    arrow_length = 1.2
    arrow_width = 0.6
    arrow_gap = 0.15  # between the point of an arrow and the cell it marks

    # how far the number on a cell is lifted off the face of that cell, in
    # cells. The two are otherwise coplanar and z-fight, which does not look
    # like noise - whole numbers simply vanish behind their own cell.
    glyph_lift = 0.02

    def __init__(self, program=None, columns=64, rows=2, cell_size=1,
                 head=0, mate=None, step_duration=0.15, start_time=3.0,
                 glyph_size=0.45, row_gap=5.0, table_gap=10.0, colors=None,
                 name="BrainFuckExtended", **kwargs):
        self.program = self.WRITE_BFF if program is None else program
        self.jumps = self._encode_jumps(self.program)
        self.columns = columns
        self.rows = rows
        self.tape_size = columns * rows
        self.cell_size = cell_size
        self.head = head
        self.mate = columns if mate is None else mate
        self.step_duration = step_duration
        self.start_time = start_time
        self.glyph_size = glyph_size
        self.row_gap = row_gap
        self.table_gap = table_gap
        overrides = colors or {}
        self.cell_colors = tuple((node_name, overrides.get(node_name, color))
                                 for node_name, color in self.CELL_COLORS)
        self.opcode_colors = tuple(
            (node_name, overrides.get(node_name, color), characters)
            for node_name, color, characters in self.OPCODE_COLORS)
        self.glyph_color = overrides.get("GlyphColor", self.GLYPH_COLOR)
        self.alphabet_color = overrides.get("AlphabetColor", self.ALPHABET_COLOR)
        self.frame_color = overrides.get("FrameColor", self.FRAME_COLOR)
        self.kwargs = kwargs
        GeometryNodesModifier.__init__(self, name=name, automatic_layout=False)

    # ----------------------------------------------------------------
    # where things are, in world units. The tape runs from x=0 to
    # x=(columns-1)*cell_size on its first line and drops by row_gap per line.
    @property
    def tape_width(self):
        return self.columns * self.cell_size

    @property
    def middle(self):
        return 0.5 * (self.columns - 1) * self.cell_size

    @property
    def table_height(self):
        """z of the line of characters in the ascii table."""
        return self.table_gap * self.cell_size

    # ----------------------------------------------------------------
    @classmethod
    def simulate(cls, program, columns=64, rows=2, head=0, mate=None):
        """Run *program* in python exactly as the graph runs it.

        :return: ``(steps, tape, head, mate)`` - the number of instructions
            executed, the tape, and where the two arrows ended up.
        """
        size = columns * rows
        mate = columns if mate is None else mate
        jumps = cls._jump_table(program)
        tape, counter, steps = [0] * size, 0, 0
        while counter < len(program):
            instruction = program[counter]
            onward = counter + 1
            if instruction == ">":
                head = min(head + 1, size - 1)
            elif instruction == "<":
                head = max(head - 1, 0)
            elif instruction == "}":
                mate = min(mate + 1, size - 1)
            elif instruction == "{":
                mate = max(mate - 1, 0)
            elif instruction == "+":
                tape[head] += 1
            elif instruction == "-":
                tape[head] -= 1
            elif instruction == ".":
                tape[mate] = tape[head]
            elif instruction == ",":
                tape[head] = tape[mate]
            elif instruction == "[" and tape[head] == 0:
                onward = jumps[counter]
            elif instruction == "]" and tape[head] != 0:
                onward = jumps[counter]
            counter, steps = onward, steps + 1
        return steps, tape, head, mate

    # ----------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        control = self._create_control_frame(tree)
        variables = self._create_variables_frame(tree)
        tape = self._create_tape_frame(tree, control)
        run = self._create_run_program_frame(tree, control, variables, tape)

        cells = self._create_cells_frame(tree, control, run)
        table = self._create_ascii_table_frame(tree, control)
        arrows = self._create_arrows_frame(tree, control, run)

        out = self.group_outputs
        out.location = (40 * 200, -2 * 200)
        join = JoinGeometry(tree, location=(38, -4))
        for piece in (cells, table, arrows):
            tree.links.new(piece, join.geometry_in)
        tree.links.new(join.geometry_out, out.inputs["Geometry"])

    # ----------------------------------------------------------------
    def _create_control_frame(self, tree):
        """``ControlParameter``: every constant of the machine."""
        x = -23.8
        control = {
            "Columns": InputInteger(tree, location=(x, 0), integer=self.columns,
                                    name="Columns"),
            "TapeSize": InputInteger(tree, location=(x, -0.8),
                                     integer=self.tape_size, name="TapeSize"),
            "CellSize": InputValue(tree, location=(x, -1.6), value=self.cell_size,
                                   name="CellSize"),
            # the square that is drawn, which is smaller than the cell it
            # stands in so that the cells do not touch - see cell_fill
            "CellDraw": InputValue(tree, location=(x, -2.0),
                                   value=self.cell_size * self.cell_fill,
                                   name="CellDraw"),
            "RowGap": InputValue(tree, location=(x, -2.4),
                                 value=self.row_gap * self.cell_size, name="RowGap"),
            "StartTime": InputValue(tree, location=(x, -3.2), value=self.start_time,
                                    name="StartTime"),
            "StepDuration": InputValue(tree, location=(x, -4.0),
                                       value=self.step_duration, name="StepDuration"),
        }

        palette = {}
        rows = ([(node_name, color) for node_name, color in self.cell_colors]
                + [(node_name, color) for node_name, color, _ in self.opcode_colors]
                + [("GlyphColor", self.glyph_color),
                   ("AlphabetColor", self.alphabet_color),
                   ("FrameColor", self.frame_color)])
        for row, (node_name, color) in enumerate(rows):
            palette[node_name] = InputMaterial(tree, location=(x, -5.4 - 0.4 * row),
                                               material=color, name=node_name,
                                               **self.kwargs)
        for source in palette.values():
            self.materials.append(source.node.material)
        control.update(palette)

        # The arrows have to fit in the gaps: the lower one points up at its
        # cell from below and the upper one points down at its from above, so
        # between the two lines there has to be room for both nose to nose, and
        # above the top line for one.
        # the point of an arrow sits arrow_gap clear of the edge of its cell,
        # and the arrow is placed by its middle, which is half its length below
        control["ArrowOffset"] = InputValue(
            tree, location=(x, -5.4 - 0.4 * len(rows)),
            value=(0.5 * self.cell_fill + self.arrow_gap
                   + 0.5 * self.arrow_length) * self.cell_size,
            name="ArrowOffset")
        control["TablePosition"] = InputVector(
            tree, location=(x, -6.2 - 0.4 * len(rows)),
            vector=Vector([self.middle, 0, self.table_height]), name="TablePosition")
        frame = Frame(tree, location=(-24, 0.6), label="ControlParameter")
        frame.add(list(control.values()))
        return control

    # ----------------------------------------------------------------
    def _create_variables_frame(self, tree):
        """``Variables``: the program, its jump table and the state seeds."""
        x = -15.8
        variables = {
            "Input": InputString(tree, location=(x, 0), string=self.program,
                                 name="Program", label="Input"),
            "Jumps": InputString(tree, location=(x, -0.8), string=self.jumps,
                                 name="JumpTable", label="Jumps"),
            "Head": InputInteger(tree, location=(x, -1.6), integer=self.head,
                                 name="LowerArrow"),
            "Mate": InputInteger(tree, location=(x, -2.4), integer=self.mate,
                                 name="UpperArrow"),
            "Counter": InputInteger(tree, location=(x, -3.2), integer=0,
                                    name="ProgramCounter"),
            # -1, so that the first step (index 0) counts as an advance
            "Step": InputInteger(tree, location=(x, -4.0), integer=-1, name="Step"),
        }
        frame = Frame(tree, location=(-16, 0.6), label="Variables")
        frame.add(list(variables.values()))
        return variables

    # ----------------------------------------------------------------
    def _create_tape_frame(self, tree, control):
        """``Tape``: ``columns * rows`` cells on ``rows`` lines, all zero.

        The points are placed in the x-z plane straight away rather than being
        laid out along x and turned afterwards, because the second line is a
        drop in *height*, not in depth.

        :return: the geometry socket of the initial tape.
        """
        index = Index(tree, location=(-9, 0.6))
        column = IntegerMath(tree, location=(-8, 1.0), operation="FLOORED_MODULO",
                             inputs0=index.std_out,
                             inputs1=control["Columns"].std_out, name="Column")
        line = IntegerMath(tree, location=(-8, 0.2), operation="DIVIDE_FLOOR",
                           inputs0=index.std_out,
                           inputs1=control["Columns"].std_out, name="Line")
        across = MathNode(tree, location=(-7, 1.0), operation="MULTIPLY",
                          inputs0=column.std_out,
                          inputs1=control["CellSize"].std_out, name="CellX")
        # lines run downwards, so cell 0 is top left and the tape reads the way
        # a page does
        down = MathNode(tree, location=(-7, 0.2), operation="MULTIPLY",
                        inputs0=line.std_out, inputs1=control["RowGap"].std_out,
                        name="LineDrop")
        height = MathNode(tree, location=(-6, 0.2), operation="MULTIPLY",
                          inputs0=down.std_out, inputs1=-1.0, name="CellZ")
        at = CombineXYZ(tree, location=(-5, 0.6), x=across.std_out, z=height.std_out,
                        name="CellPosition")
        points = Points(tree, location=(-4, 0.6), count=self.tape_size,
                        position=at.std_out)
        # the attribute has to exist from the first frame on, otherwise the
        # "Sample Index" in the automaton has nothing to read
        zeros = StoredNamedAttribute(tree, location=(-3, 0.6), data_type="INT",
                                     domain="POINT", name="Value", value=0,
                                     label="ClearTape")
        create_geometry_line(tree, [points, zeros])
        frame = Frame(tree, location=(-9.4, 1.8), label="Tape")
        frame.add([index, column, line, across, down, height, at, points, zeros])
        return zeros.geometry_out

    # ----------------------------------------------------------------
    def _create_run_program_frame(self, tree, control, variables, tape):
        """``RunProgram``: the simulation zone - the clock and the counter.

        The clock is the one of :class:`SimpleBrainFuckModifier`; what differs
        is the state it carries, which is two arrows rather than one head and a
        printed string.

        :return: ``{name: socket}`` of the state as it leaves the zone.
        """
        zone = Simulation(tree, location=(2, 5), node_width=20, geometry=tape)
        sim_in, sim_out = zone.simulation_input, zone.simulation_output
        for socket_type, socket_name, initial in (
                ("FLOAT", "StartTime", control["StartTime"].std_out),
                ("INT", "Step", variables["Step"].std_out),
                ("INT", "LowerArrow", variables["Head"].std_out),
                ("INT", "UpperArrow", variables["Mate"].std_out),
                ("INT", "Counter", variables["Counter"].std_out),
                ("FLOAT", "Time", 0.0)):
            zone.add_socket(socket_type=socket_type, name=socket_name, value=initial)

        time = MathNode(tree, location=(3.2, 6.4), operation="ADD",
                        inputs0=sim_in.outputs["Delta Time"],
                        inputs1=sim_in.outputs["Time"], name="Clock")
        since = MathNode(tree, location=(4.4, 6.4), operation="SUBTRACT",
                         inputs0=time.std_out, inputs1=sim_in.outputs["StartTime"],
                         name="SinceStart")
        scaled = MathNode(tree, location=(5.6, 6.4), operation="DIVIDE",
                          inputs0=since.std_out,
                          inputs1=control["StepDuration"].std_out, name="InSteps")
        waiting = MathNode(tree, location=(6.8, 6.4), operation="MAXIMUM",
                           inputs0=scaled.std_out, inputs1=-1.0, name="NotBeforeStart")
        step = MathNode(tree, location=(8.0, 6.4), operation="FLOOR",
                        inputs0=waiting.std_out, name="StepIndex")
        advance = CompareNode(tree, location=(9.2, 6.4), operation="GREATER_THAN",
                              data_type="INT", inputs0=step.std_out,
                              inputs1=sim_in.outputs["Step"], name="Advance")

        program = variables["Input"].std_out
        current = SliceString(tree, location=(3.2, 4.6), string=program,
                              position=sim_in.outputs["Counter"], length=1,
                              name="Instruction")
        opcode = CharToAscii(tree, location=(4.4, 4.6), char=current.std_out)
        length = StringLength(tree, location=(3.2, 3.6), string=program,
                              name="ProgramLength")
        running = CompareNode(tree, location=(4.4, 3.6), operation="LESS_THAN",
                              data_type="INT", inputs0=sim_in.outputs["Counter"],
                              inputs1=length.std_out, name="NotHalted")
        fire = BooleanMath(tree, location=(10.4, 6.4), operation="AND",
                           inputs0=advance.std_out, inputs1=running.std_out,
                           name="ExecuteNow")

        code_in = Reroute(tree, location=(11.6, 4.6), ins=opcode.std_out, name="Opcode")
        fire_in = Reroute(tree, location=(11.6, 4.2), ins=fire.std_out, name="Fire")
        head_in = Reroute(tree, location=(11.6, 3.8),
                          ins=sim_in.outputs["LowerArrow"], name="Lower")
        mate_in = Reroute(tree, location=(11.6, 3.4),
                          ins=sim_in.outputs["UpperArrow"], name="Upper")
        step_in = Reroute(tree, location=(11.6, 3.0), ins=sim_in.outputs["Counter"],
                          name="Counter")

        head, mate, tape_out, counter = self._create_automaton_frame(
            tree, control, variables, sim_in, code_in.std_out, fire_in.std_out,
            head_in.std_out, mate_in.std_out, step_in.std_out)

        for socket, name in ((time.std_out, "Time"), (step.std_out, "Step"),
                             (sim_in.outputs["StartTime"], "StartTime"),
                             (counter, "Counter"), (head, "LowerArrow"),
                             (mate, "UpperArrow")):
            tree.links.new(socket, sim_out.inputs[name])
        tree.links.new(tape_out, sim_out.inputs["Geometry"])

        frame = Frame(tree, location=(1.6, 7.4), label="RunProgram")
        frame.add([zone, time, since, scaled, waiting, step, advance, running, fire,
                   current, opcode, length,
                   code_in, fire_in, head_in, mate_in, step_in])
        return {name: sim_out.outputs[name] for name in
                ("Geometry", "Step", "LowerArrow", "UpperArrow", "Counter")}

    # ----------------------------------------------------------------
    def _create_automaton_frame(self, tree, control, variables, sim_in, opcode,
                                fire, head, mate, counter):
        """``Automaton``: what the ten instructions do.

        :return: ``(lower, upper, tape, counter)`` sockets for the state.
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

        last = keep(IntegerMath(tree, location=(16.0, 5.4), operation="SUBTRACT",
                                inputs0=control["TapeSize"].std_out, inputs1=1,
                                name="LastCell"))

        def walk(where, forward, backward, row, label):
            """An arrow moved one cell and kept on the tape."""
            ahead = keep(IntegerMath(tree, location=(16.0, row), operation="ADD",
                                     inputs0=where, inputs1=forward,
                                     name=label + "Right"))
            back = keep(IntegerMath(tree, location=(17.0, row), operation="SUBTRACT",
                                    inputs0=ahead.std_out, inputs1=backward,
                                    name=label + "Left"))
            capped = keep(IntegerMath(tree, location=(18.0, row), operation="MINIMUM",
                                      inputs0=back.std_out, inputs1=last.std_out,
                                      name=label + "NotPastTheEnd"))
            return keep(IntegerMath(tree, location=(19.0, row), operation="MAXIMUM",
                                    inputs0=capped.std_out, inputs1=0,
                                    name=label + "NotBeforeStart")).std_out

        # --- the two arrows --------------------------------------------
        lower = walk(head,
                     step_of(decodes(self.RIGHT, 4.2, "Right"), 4.2, "StepRight"),
                     step_of(decodes(self.LEFT, 3.4, "Left"), 3.4, "StepLeft"),
                     3.8, "Lower")
        upper = walk(mate,
                     step_of(decodes(self.BRACE_RIGHT, 2.6, "BraceRight"), 2.6,
                             "SlideRight"),
                     step_of(decodes(self.BRACE_LEFT, 1.8, "BraceLeft"), 1.8,
                             "SlideLeft"),
                     2.2, "Upper")

        # --- what the two cells hold ------------------------------------
        stored = NamedAttribute(tree, location=(12.4, 0.6), data_type="INT",
                                name="Value")
        cell = SampleIndex(tree, location=(13.6, 1.0), data_type="INT", domain="POINT",
                           geometry=sim_in.outputs["Geometry"], value=stored.std_out,
                           index=head, name="CellUnderLower")
        other = SampleIndex(tree, location=(13.6, 0.0), data_type="INT", domain="POINT",
                            geometry=sim_in.outputs["Geometry"], value=stored.std_out,
                            index=mate, name="CellUnderUpper")

        plus = step_of(decodes(self.PLUS, -0.8, "Plus"), -0.8, "Increment")
        minus = step_of(decodes(self.MINUS, -1.6, "Minus"), -1.6, "Decrement")
        raised = IntegerMath(tree, location=(16.0, 0.6), operation="ADD",
                             inputs0=cell.std_out, inputs1=plus, name="CellPlus")
        lowered = IntegerMath(tree, location=(17.0, 0.6), operation="SUBTRACT",
                              inputs0=raised.std_out, inputs1=minus, name="CellMinus")
        # "," overrides the arithmetic rather than adding to it: it is the one
        # instruction that puts something into the lower cell from outside
        reads = decodes(self.COMMA, -2.4, "Comma")
        writes = decodes(self.DOT, -3.2, "Dot")
        fetched = Switch(tree, location=(18.0, 0.6), input_type="INT", switch=reads,
                         false=lowered.std_out, true=other.std_out,
                         name="NewLowerValue")

        here = Index(tree, location=(16.0, -0.4))
        on_lower = CompareNode(tree, location=(17.0, -0.4), operation="EQUAL",
                               data_type="INT", inputs0=here.std_out, inputs1=head,
                               name="AtTheLowerArrow", hide=True)
        write_lower = StoredNamedAttribute(tree, location=(19.4, 0.6), data_type="INT",
                                           domain="POINT", name="Value",
                                           selection=on_lower.std_out,
                                           value=fetched.std_out, label="WriteLower")
        tree.links.new(sim_in.outputs["Geometry"], write_lower.geometry_in)

        # The upper cell is only ever written by ".", and the selection says so
        # rather than the value: without the "AND" a "+" on a frame where both
        # arrows are on the same cell would be undone by this node writing the
        # value it sampled before the increment.
        on_upper = CompareNode(tree, location=(17.0, -1.2), operation="EQUAL",
                               data_type="INT", inputs0=here.std_out, inputs1=mate,
                               name="AtTheUpperArrow", hide=True)
        copies = BooleanMath(tree, location=(18.0, -1.2), operation="AND",
                             inputs0=on_upper.std_out, inputs1=writes,
                             name="CopyToUpper", hide=True)
        write_upper = StoredNamedAttribute(tree, location=(20.4, 0.6), data_type="INT",
                                           domain="POINT", name="Value",
                                           selection=copies.std_out,
                                           value=cell.std_out, label="WriteUpper")
        create_geometry_line(tree, [write_lower, write_upper])

        # --- the loop, and where the counter goes next -------------------
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
        entry = SliceString(tree, location=(12.4, -5.8),
                            string=variables["Jumps"].std_out, position=counter,
                            length=1, name="JumpEntry")
        encoded = CharToAscii(tree, location=(13.6, -5.8), char=entry.std_out,
                              name="JumpCode")
        target = IntegerMath(tree, location=(14.8, -5.8), operation="SUBTRACT",
                             inputs0=encoded.std_out, inputs1=self.JUMP_ORIGIN,
                             name="JumpTarget")
        onward = IntegerMath(tree, location=(14.8, -6.6), operation="ADD",
                             inputs0=counter, inputs1=1, name="NextInstruction")
        jumped = Switch(tree, location=(18.0, -5.8), input_type="INT",
                        switch=jumping.std_out, false=onward.std_out,
                        true=target.std_out, name="CounterAfterStep")
        moved = Switch(tree, location=(19.0, -5.8), input_type="INT",
                       switch=fire, false=counter, true=jumped.std_out,
                       name="NewCounter")

        frame = Frame(tree, location=(12.0, 6.0), label="Automaton")
        frame.add(built + [stored, cell, other, raised, lowered, fetched, here,
                           on_lower, write_lower, on_upper, copies, write_upper,
                           empty, filled, skips, repeats, jumping,
                           entry, encoded, target, onward, jumped, moved])
        return lower, upper, write_upper.geometry_out, moved.std_out

    # ----------------------------------------------------------------
    def _create_cells_frame(self, tree, control, run):
        """``Cells``: the tape as it looks, coloured by what is on it.

        The cell square is turned upright once, before it is instanced, rather
        than the whole tape being turned afterwards - the tape is already in
        the plane it is drawn in, and turning it would fold the second line
        away from the camera instead of below the first.

        :return: the geometry socket of the finished tape.
        """
        tape = run["Geometry"]
        quad = Quadrilateral(tree, location=(26, 2), mode="RECTANGLE",
                             width=control["CellDraw"].std_out,
                             height=control["CellDraw"].std_out)
        fill = FillCurve(tree, location=(27, 2), mode="N-gons")
        upright = TransformGeometry(tree, location=(28, 2), rotation=[pi / 2, 0, 0],
                                    name="StandCellUp")
        create_geometry_line(tree, [quad, fill, upright])
        instances = InstanceOnPoints(tree, location=(29, 2.6), points=tape,
                                     instance=upright.geometry_out)
        realize = RealizeInstances(tree, location=(30, 2.6))

        value = NamedAttribute(tree, location=(29, 1.2), data_type="INT", name="Value")
        here = Index(tree, location=(29, 0.6))
        holds = CompareNode(tree, location=(30, 1.6), operation="NOT_EQUAL",
                            data_type="INT", inputs0=value.std_out, inputs1=0,
                            name="CellHoldsAValue", hide=True)
        under = CompareNode(tree, location=(30, 1.0), operation="EQUAL",
                            data_type="INT", inputs0=here.std_out,
                            inputs1=run["LowerArrow"], name="CellUnderLower",
                            hide=True)
        over = CompareNode(tree, location=(30, 0.4), operation="EQUAL",
                           data_type="INT", inputs0=here.std_out,
                           inputs1=run["UpperArrow"], name="CellUnderUpper",
                           hide=True)
        selections = (None, holds.std_out, under.std_out, over.std_out)

        painters = [SetMaterial(tree, location=(31 + column, 2.6), selection=selection,
                                material=control[node_name].std_out,
                                name="Paint" + node_name)
                    for column, ((node_name, _), selection)
                    in enumerate(zip(self.cell_colors, selections))]
        create_geometry_line(tree, [instances, realize] + painters)

        numbers = self._create_cell_values(tree, control, tape)
        joined = JoinGeometry(tree, location=(36, 2.6))
        tree.links.new(painters[-1].geometry_out, joined.geometry_in)
        tree.links.new(numbers, joined.geometry_in)

        frame = Frame(tree, location=(25.6, 3.4), label="Cells")
        frame.add([quad, fill, upright, instances, realize, value, here, holds,
                   under, over, joined] + painters)
        return joined.geometry_out

    # ----------------------------------------------------------------
    def _create_cell_values(self, tree, control, tape):
        """``CellValues``: the number a cell holds, written on it.

        An empty cell is left empty rather than being written with a nought:
        on a tape of 128 cells of which a handful are ever used, a wall of
        noughts is all one would see.

        :return: the geometry socket of the numbers.
        """
        value = NamedAttribute(tree, location=(26, -2), data_type="INT", name="Value")
        position = Position(tree, location=(26, -2.6))
        zone = ForEachZone(tree, location=(27, -1.4), domain="POINT", node_width=7,
                           geometry=tape)
        zone.add_socket(socket_type="INT", name="Value", value=value.std_out,
                        for_input=True)
        zone.add_socket(socket_type="VECTOR", name="Location", value=position.std_out,
                        for_input=True)

        digits = ValueToString(tree, location=(28, -0.8), data_type="INT",
                               value=zone.foreach_input.outputs["Value"],
                               name="CellValue")
        holds = CompareNode(tree, location=(28, -1.6), operation="NOT_EQUAL",
                            data_type="INT",
                            inputs0=zone.foreach_input.outputs["Value"], inputs1=0,
                            name="CellHoldsAValue", hide=True)
        shown = Switch(tree, location=(29, -0.8), input_type="STRING",
                       switch=holds.std_out, false="", true=digits.std_out,
                       name="ShownValue")
        size = MathNode(tree, location=(28, -2.4), operation="MULTIPLY",
                        inputs0=control["CellSize"].std_out, inputs1=self.glyph_size,
                        name="NumberSize")
        curves = StringToCurves(tree, location=(30, -1.4), string=shown.std_out,
                                size=size.std_out, align_x="CENTER", align_y="MIDDLE")
        realize = RealizeInstances(tree, location=(31, -1.4))
        fill = FillCurve(tree, location=(32, -1.4), mode="N-gons")
        painted = SetMaterial(tree, location=(33, -1.4),
                              material=control["GlyphColor"].std_out,
                              name="PaintNumber")
        lift = VectorMath(tree, location=(33, -2.4), operation="ADD",
                          inputs0=zone.foreach_input.outputs["Location"],
                          inputs1=[0, -self.glyph_lift * self.cell_size, 0],
                          name="LiftNumber")
        placed = TransformGeometry(tree, location=(34, -1.4),
                                   translation=lift.std_out,
                                   rotation=[pi / 2, 0, 0], name="PlaceNumber")
        zone.create_geometry_line([realize, fill, painted, placed],
                                  ins=curves.geometry_out)

        frame = Frame(tree, location=(25.6, -0.6), label="CellValues")
        frame.add([value, position, zone, digits, holds, shown, size, curves,
                   realize, fill, painted, lift, placed])
        return zone.geometry_out

    # ----------------------------------------------------------------
    def _create_ascii_table_frame(self, tree, control):
        """``AsciiTable``: the 95 printable characters, framed.

        A repeat zone walks the codes. Each one is turned into its character,
        and the ten that the machine reads as instructions are painted in the
        colour of their family and drawn twice - filled, and with their outline
        traced - which is what makes them bold. The code itself is written
        above every tenth entry.

        :return: the geometry socket of the table.
        """
        count = self.LAST_PRINTABLE - self.FIRST_PRINTABLE + 1
        spacing = self.tape_width / count
        zone = RepeatZone(tree, location=(-13, 16), node_width=10, iterations=count)
        step = zone.iteration

        origin = SeparateXYZ(tree, location=(-12, 18.2),
                             vector=control["TablePosition"].std_out)
        column = MathNode(tree, location=(-12, 17.4), operation="MULTIPLY",
                          inputs0=step, inputs1=spacing, name="Column")
        start = MathNode(tree, location=(-11, 18.2), operation="SUBTRACT",
                         inputs0=origin.x, inputs1=0.5 * (count - 1) * spacing,
                         name="TableStart")
        across = MathNode(tree, location=(-10, 17.4), operation="ADD",
                          inputs0=start.std_out, inputs1=column.std_out,
                          name="AtColumn")
        letter_at = CombineXYZ(tree, location=(-9, 17.4), x=across.std_out,
                               y=origin.y, z=origin.z, name="LetterPosition")
        above = MathNode(tree, location=(-10, 16.6), operation="ADD",
                         inputs0=origin.z, inputs1=self.table_line_gap,
                         name="LabelLine")
        label_at = CombineXYZ(tree, location=(-9, 16.6), x=across.std_out,
                              y=origin.y, z=above.std_out, name="LabelPosition")

        # the character of this code. The table starts at the first printable
        # code, so the code is the step plus that offset
        code = IntegerMath(tree, location=(-12, 15.8), operation="ADD", inputs0=step,
                           inputs1=self.FIRST_PRINTABLE, name="Code")
        table = InputString(tree, location=(-14.4, 15.8),
                            string="".join(chr(value) for value
                                           in range(self.FIRST_PRINTABLE,
                                                    self.LAST_PRINTABLE + 1)),
                            name="Printable")
        letter = SliceString(tree, location=(-11, 15.8), string=table.std_out,
                             position=step, length=1, name="Letter")
        curves = StringToCurves(tree, location=(-10, 15.8), string=letter.std_out,
                                size=self.table_glyph_size, align_x="CENTER",
                                align_y="MIDDLE", hide=True)
        realize = RealizeInstances(tree, location=(-9, 15.8))
        fill = FillCurve(tree, location=(-8, 15.8), mode="N-gons")
        create_geometry_line(tree, [realize, fill], ins=curves.geometry_out)
        # bold, by tracing the outline of the glyph as well as filling it. A
        # second font would be the other way, and would have to be found.
        stroke = CurveWireFrame(tree, location=(-8, 15.0),
                                radius=self.bold_weight * self.table_glyph_size,
                                resolution=4, geometry=realize.geometry_out)
        weight = JoinGeometry(tree, location=(-7, 15.4))
        for piece in (fill.geometry_out, stroke.geometry_out):
            tree.links.new(piece, weight.geometry_in)

        # "is this character one of ours". Find in String counts the character
        # in the instruction set, which is 1 for an instruction and 0 for
        # everything else - one node instead of ten comparisons
        instructions = "".join(characters
                               for _, _, characters in self.opcode_colors)
        found = FindInString(tree, location=(-12, 14.2), string=instructions,
                             search=letter.std_out, name="AmongTheInstructions",
                             hide=True)
        is_op = CompareNode(tree, location=(-11, 14.2), operation="NOT_EQUAL",
                            data_type="INT", inputs0=found.count_out,
                            inputs1=0, name="IsAnInstruction", hide=True)
        bold = Switch(tree, location=(-6, 15.8), input_type="GEOMETRY",
                      switch=is_op.std_out, false=fill.geometry_out,
                      true=weight.geometry_out, name="BoldIfInstruction")

        painters = [SetMaterial(tree, location=(-5, 15.8),
                                material=control["AlphabetColor"].std_out,
                                name="PaintAlphabet")]
        for row, (node_name, _, characters) in enumerate(self.opcode_colors):
            selection = self._is_one_of(tree, characters, code.std_out,
                                        (-11, 13.4 - 0.6 * row))
            painters.append(SetMaterial(tree, location=(-4 + row, 15.8),
                                        selection=selection,
                                        material=control[node_name].std_out,
                                        name="Paint" + node_name))
        place = TransformGeometry(tree, location=(-4 + len(self.opcode_colors), 15.8),
                                  translation=letter_at.std_out,
                                  rotation=[pi / 2, 0, 0], name="PlaceLetter")
        create_geometry_line(tree, [bold] + painters + [place])

        # the code, above every tenth character - 95 three-digit numbers across
        # the width of the tape would be a few pixels each
        number = ValueToString(tree, location=(-11, 16.6), data_type="INT",
                               value=code.std_out, name="CodeLabel")
        tick = IntegerMath(tree, location=(-12, 17.0), operation="FLOORED_MODULO",
                           inputs0=code.std_out, inputs1=self.table_label_every,
                           name="EveryTenth")
        labelled = CompareNode(tree, location=(-11, 17.0), operation="EQUAL",
                               data_type="INT", inputs0=tick.std_out, inputs1=0,
                               name="IsATick", hide=True)
        label_curves = StringToCurves(tree, location=(-10, 16.2),
                                      string=number.std_out,
                                      size=self.table_label_size, align_x="CENTER",
                                      align_y="MIDDLE", hide=True)
        label_realize = RealizeInstances(tree, location=(-9, 16.2))
        label_fill = FillCurve(tree, location=(-8, 16.2), mode="N-gons")
        label_paint = SetMaterial(tree, location=(-7, 16.2),
                                  material=control["AlphabetColor"].std_out,
                                  name="PaintCode")
        label_place = TransformGeometry(tree, location=(-6, 16.2),
                                        translation=label_at.std_out,
                                        rotation=[pi / 2, 0, 0], name="PlaceCode")
        create_geometry_line(tree, [label_realize, label_fill, label_paint,
                                    label_place], ins=label_curves.geometry_out)
        ticked = Switch(tree, location=(-5, 16.2), input_type="GEOMETRY",
                        switch=labelled.std_out, true=label_place.geometry_out,
                        name="OnlyEveryTenth")

        entry = JoinGeometry(tree, location=(-3, 16))
        for piece in (place.geometry_out, ticked.geometry_out):
            tree.links.new(piece, entry.geometry_in)
        grown = JoinGeometry(tree, location=(-2.4, 16))
        tree.links.new(entry.geometry_out, grown.geometry_in)
        tree.links.new(zone.repeat_input.outputs["Geometry"], grown.geometry_in)
        tree.links.new(grown.geometry_out, zone.repeat_output.inputs["Geometry"])

        box = self._create_table_frame(tree, control, zone.geometry_out)
        joined = JoinGeometry(tree, location=(6, 16))
        for piece in (zone.geometry_out, box):
            tree.links.new(piece, joined.geometry_in)

        frame = Frame(tree, location=(-14.6, 19.0), label="AsciiTable")
        frame.add([zone, origin, column, start, across, letter_at, above, label_at,
                   code, table, letter, curves, realize, fill, stroke, weight,
                   found, is_op, bold, place, number, tick, labelled, label_curves,
                   label_realize, label_fill, label_paint, label_place, ticked,
                   entry, grown, joined] + painters)
        return joined.geometry_out

    # ----------------------------------------------------------------
    def _is_one_of(self, tree, characters, code, location):
        """``True`` where *code* is the code of one of *characters*.

        One ``Compare`` per character, ``OR``-ed together - the sets are one or
        two characters long, so this stays small.
        """
        result = None
        for offset, character in enumerate(characters):
            same = CompareNode(tree, location=(location[0], location[1] - 0.3 * offset),
                               operation="EQUAL", data_type="INT", inputs0=code,
                               inputs1=ord(character), name="Is" + character, hide=True)
            if result is None:
                result = same.std_out
            else:
                result = BooleanMath(tree, location=(location[0] + 1, location[1]),
                                     operation="OR", inputs0=result,
                                     inputs1=same.std_out, hide=True).std_out
        return result

    # ----------------------------------------------------------------
    def _create_arrows_frame(self, tree, control, run):
        """``Arrows``: the two heads, one under the tape and one over it.

        Both are placed from the position of the cell they are on, read off the
        tape with ``Sample Index`` - which is the only thing that keeps them on
        the right line when the tape is folded into two.

        :return: the geometry socket of the two arrows.
        """
        at = Position(tree, location=(26, -8))
        pieces, arrows = [], []
        for row, (label, index, colour, sign) in enumerate((
                ("LowerArrow", run["LowerArrow"], "LowerColor", -1.0),
                ("UpperArrow", run["UpperArrow"], "UpperColor", 1.0))):
            y = -8 - 4 * row
            spot = SampleIndex(tree, location=(27, y), data_type="FLOAT_VECTOR",
                               domain="POINT", geometry=run["Geometry"],
                               value=at.std_out, index=index,
                               name="CellOf" + label)
            along = SeparateXYZ(tree, location=(28, y), vector=spot.std_out)
            # the lower arrow hangs below its cell and the upper one above it
            drop = MathNode(tree, location=(28, y - 0.8), operation="MULTIPLY",
                            inputs0=control["ArrowOffset"].std_out, inputs1=sign,
                            name="Offset" + label)
            height = MathNode(tree, location=(29, y - 0.8), operation="ADD",
                              inputs0=along.z, inputs1=drop.std_out,
                              name="Height" + label)
            where = CombineXYZ(tree, location=(30, y), x=along.x, z=height.std_out,
                               name="Place" + label)
            tip = ConeMesh(tree, location=(27, y - 1.6), vertices=32, radius_top=0,
                           radius_bottom=0.5 * self.arrow_width * self.cell_size,
                           depth=0.5 * self.arrow_length * self.cell_size)
            stem = CylinderMesh(tree, location=(27, y - 2.4), vertices=32,
                                radius=0.25 * self.arrow_width * self.cell_size,
                                depth=0.5 * self.arrow_length * self.cell_size)
            below = TransformGeometry(tree, location=(28, y - 2.4),
                                      translation=[0, 0,
                                                   -0.25 * self.arrow_length
                                                   * self.cell_size],
                                      name="Stem" + label)
            create_geometry_line(tree, [stem, below])
            body = JoinGeometry(tree, location=(29, y - 1.6))
            for piece in (tip.geometry_out, below.geometry_out):
                tree.links.new(piece, body.geometry_in)
            # the cone is born pointing along +z, which is right for the arrow
            # that points up at its cell from below; the other one is turned over
            turned = TransformGeometry(tree, location=(30, y - 1.6),
                                       rotation=[0, 0, 0] if sign < 0
                                       else [pi, 0, 0], name="Turn" + label)
            put = TransformGeometry(tree, location=(31, y), translation=where.std_out,
                                    name="Put" + label)
            painted = SetMaterial(tree, location=(32, y),
                                  material=control[colour].std_out,
                                  name="Paint" + label)
            create_geometry_line(tree, [body, turned, put, painted])
            pieces += [spot, along, drop, height, where, tip, stem, below, body,
                       turned, put, painted]
            arrows.append(painted)

        joined = JoinGeometry(tree, location=(34, -10))
        for arrow in arrows:
            tree.links.new(arrow.geometry_out, joined.geometry_in)

        frame = Frame(tree, location=(25.6, -7.2), label="Arrows")
        frame.add(pieces + [at, joined])
        return joined.geometry_out
