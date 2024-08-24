#!/usr/bin/env python

from manim_imports_ext import *
import operator as op


class BehindTheScenes(GraphScene):
    def __init__(self, **kwargs):
        GraphScene.__init__(self, x_min=-3,
                            x_max=3,
                            y_min=-3,
                            y_max=3,
                            y_axis_height=6,
                            x_axis_width=6,
                            y_axis_label="",
                            x_axis_label="",
                            graph_origin=RIGHT * 3.5 + DOWN * 0.5,
                            **kwargs)

    def construct(self):
        objects = []
        title = Title("Behind the scenes")
        title.set_color(YELLOW)
        title.to_edge(UP)
        self.play(Write(title))

        r3 = np.sqrt(3)

        p_vec = TexMobject(r"\vec P = \frac{1}{2}\left(\begin{array}{c}\sqrt{3}\\-1\end{array}\right)")
        p_vec.set_color(YELLOW)
        p_vec.scale(0.6)
        p_vec.next_to(title, DOWN)
        p_vec.to_edge(LEFT)

        self.play(Write(p_vec))

        self.setup_axes()

        p = Dot()
        objects.append(p)
        p.move_to(self.coords_to_point(0.5 * r3, -0.5))
        p.set_color(YELLOW)
        self.play(ShowCreation(p))

        line_c = Line(3 * LEFT, 3 * RIGHT)
        line_c.move_to(self.coords_to_point(0, 0))
        line_c.set_color(MARINE)
        line_c.set_style(MARINE, 1, MARINE, 10, 1)

        mat_c = TexMobject(r"M_c = \left(\begin{array}{c c} 1 & 0\\0 & -1\end{array}\right)")
        mat_c.set_color(MARINE)
        mat_c.scale(0.7)
        mat_c.next_to(title, DOWN)
        mat_c.align_to(title, RIGHT)
        mat_c.shift(0.5 * RIGHT)

        self.play(ShowCreation(line_c), Write(mat_c))
        self.wait(3)

        p_prime = TexMobject(r"\vec P'", "=", "M_c", r"\cdot", r"\vec P", "=", r"\frac{1}{2}\left(\begin{array}{"
                                                                               r"c}\sqrt{3}\\1\end{array}\right)")
        p_prime[0].set_color(MARINE)
        p_prime[2].set_color(MARINE)
        p_prime[4].set_color(YELLOW)
        p_prime[6].set_color(MARINE)

        p_prime.next_to(p_vec, DOWN)
        p_prime.align_to(p_vec, LEFT)

        pp = Dot()
        objects.append(pp)
        pp.move_to(self.coords_to_point(0.5 * r3, 0.5))
        pp.set_color(MARINE)

        self.play(Write(p_prime), ShowCreation(pp))
        self.wait(3)

        line_b = Line(3 * LEFT, 3 * RIGHT)
        line_b.move_to(self.coords_to_point(0, 0))
        line_b.set_color(GREEN)
        line_b.set_style(GREEN, 1, GREEN, 10, 1)
        line_b.rotate(-np.pi / 6)

        mat_b = TexMobject("M_b =", r"\frac{1}{2}\left(\begin{array}{c c} 1 & -\sqrt{3}\\-\sqrt{3} & -1\end{"
                                    r"array}\right)")
        mat_b.set_color(GREEN)
        mat_b.scale(0.7)
        mat_b.to_edge(DOWN)
        mat_b.align_to(mat_c, LEFT)
        mat_b[1].scale(0.7)
        mat_b[1].shift(0.5 * LEFT)

        self.play(ShowCreation(line_b), Write(mat_b))
        self.wait(3)

        p_prime2 = TexMobject(r"\vec P''", "=", "M_b", r"\cdot", r"\vec P'", "=", r"\left(\begin{array}{c}0\\-1\end{"
                                                                                  r"array}\right)")
        p_prime2[0].set_color(CYAN)
        p_prime2[2].set_color(GREEN)
        p_prime2[4].set_color(MARINE)
        p_prime2[6].set_color(CYAN)

        p_prime2.next_to(p_prime, DOWN)
        p_prime2.align_to(p_prime, LEFT)

        ppp = Dot()
        objects.append(ppp)
        ppp.move_to(self.coords_to_point(0, -1))
        ppp.set_color(CYAN)

        self.play(Write(p_prime2), ShowCreation(ppp))
        self.wait(3)

        prod = VGroup(p_prime2[2], p_prime2[3], p_prime2[4])
        prod2 = TexMobject(r"(M_b\cdot M_c)", r"\cdot", r"\vec P")
        prod2[0].set_color(CYAN)
        prod2[2].set_color(YELLOW)
        prod2.scale(0.6)
        prod2.move_to(prod)

        p1 = TexMobject(r"\vec P_1")
        p1.set_color(CYAN)
        p1.move_to(p_prime2[0])

        self.play(Transform(prod, prod2), Transform(p_prime2[0], p1))
        self.wait(3)

        p_prime2.generate_target()
        p_prime2.target.scale(0.7)
        p_prime2.target.next_to(p_vec, DOWN)
        p_prime2.target.to_edge(LEFT, buff=SMALL_BUFF)
        self.play(Uncreate(p_prime))
        self.play(p_prime2.next_to, p_vec, DOWN, MoveToTarget(p_prime2))

        self.wait(3)

        lines = [p_prime2]
        rhs = [r"\frac{1}{2}\left(\begin{array}{c}-\sqrt{3}\\-1\end{array}\right)",
               r"\frac{1}{2}\left(\begin{array}{c}-\sqrt{3}\\1\end{array}\right)",
               r"\phantom{\frac{1}{2}}\left(\begin{array}{c}0\\1\end{array}\right)",
               r"\frac{1}{2}\left(\begin{array}{c}\sqrt{3}\\1\end{array}\right)",
               r"\frac{1}{2}\left(\begin{array}{c}\sqrt{3}\\-1\end{array}\right)"]
        coords = [-r3, -1, -r3, 1, 0, 2, r3, 1, r3, -1]

        for i in range(2, 7):
            text = TexMobject(r"\vec P_", i, "=", r"(M_b\cdot M_c)^", i, r"\cdot", r"\vec P", "=", rhs[i - 2])
            text[0].set_color(CYAN)
            text[1].set_color(CYAN)
            text[3].set_color(CYAN)
            text[6].set_color(YELLOW)
            text[8].set_color(CYAN)
            text.scale(0.6)
            text.next_to(lines[i - 2], DOWN)
            text.to_edge(LEFT, buff=SMALL_BUFF)
            lines.append(text)
            dot = Dot()
            objects.append(dot)
            dot.move_to(self.coords_to_point(0.5 * coords[2 * (i - 2)], 0.5 * coords[2 * (i - 2) + 1]))
            dot.set_color(CYAN)
            self.play(Write(text), ShowCreation(dot))
            self.wait()

        self.wait(3)

        hexagon = Polygon(self.coords_to_point(0.5 * r3, -0.5), self.coords_to_point(0, -1),
                          self.coords_to_point(-0.5 * r3, -0.5), self.coords_to_point(-0.5 * r3, 0.5),
                          self.coords_to_point(0, 1), self.coords_to_point(0.5 * r3, 0.5),
                          self.coords_to_point(0.5 * r3, -0.5),
                          stroke_width=5, fill_opacity=0.5)
        hexagon.set_color(CYAN)
        objects.append(hexagon)

        self.play(ShowCreation(hexagon))
        self.wait(3)

        mat_a = TexMobject("M_a =", r"\left(\begin{array}{c c} -1 & 0\\ 0 & 1\end{array}\right)")
        mat_a.set_color(RED)
        mat_a.scale(0.7)
        mat_a.next_to(mat_c, LEFT)

        line_a = Line(2 * UP, 2 * DOWN)
        line_a.move_to(self.coords_to_point(0.5 * r3, 0))
        line_a.set_color(RED)
        line_a.set_style(RED, 1, RED, 10, 1)
        self.play(Uncreate(p_vec), ShowCreation(line_a), Write(mat_a),
                  *[Uncreate(lines[i]) for i in range(0, len(lines))])
        self.wait(3)

        p_red = TexMobject(r"\vec P_i'", "=", "M_a", r"\cdot", r"\vec P_i", "+",
                           r"\left(\begin{array}{c} \sqrt{3}\\0\end{array}\right)")
        p_red[0].set_color(RED)
        p_red[2].set_color(RED)
        p_red[4].set_color(CYAN)
        p_red[6].scale(0.7)
        p_red.next_to(title, DOWN)
        p_red.to_edge(LEFT)

        coords2 = [[1.7320508075688772, -1.], [2.598076211353316, -0.5], [2.598076211353316, 0.5],
                   [1.7320508075688772, 1.], [0.8660254037844386, 0.5], [0.8660254037844386, -0.5]]

        hexagon2 = Polygon(self.coords_to_point(coords2[0][0], coords2[0][1]),
                           self.coords_to_point(coords2[1][0], coords2[1][1]),
                           self.coords_to_point(coords2[2][0], coords2[2][1]),
                           self.coords_to_point(coords2[3][0], coords2[3][1]),
                           self.coords_to_point(coords2[4][0], coords2[4][1]),
                           self.coords_to_point(coords2[5][0], coords2[5][1]),
                           self.coords_to_point(coords2[0][0], coords2[0][1]), stroke_width=5, fill_opacity=0.5)
        hexagon2.set_color(RED)
        objects.append(hexagon2)

        self.play(ShowCreation(hexagon2), Write(p_red))
        self.wait(3)

        coords3 = [[[0., -1.9999999999999998], [0.8660254037844387, -2.5], [1.7320508075688772, -2.],
                    [1.7320508075688772, -0.9999999999999998], [0.8660254037844386, -0.4999999999999999],
                    [0., -0.9999999999999999]],
                   [[-1.7320508075688772, -0.9999999999999998], [-1.7320508075688772, -2.], [-0.8660254037844387, -2.5],
                    [0., -1.9999999999999998], [0., -0.9999999999999999], [-0.8660254037844386, -0.4999999999999999]],
                   [[-1.7320508075688772, 1.], [-2.598076211353316, 0.5], [-2.598076211353316, -0.5],
                    [-1.7320508075688772, -1.], [-0.8660254037844386, -0.5], [-0.8660254037844386, 0.5]],
                   [[0., 1.9999999999999998], [-0.8660254037844387, 2.5], [-1.7320508075688772, 2.],
                    [-1.7320508075688772, 0.9999999999999998], [-0.8660254037844386, 0.4999999999999999],
                    [0., 0.9999999999999999]],
                   [[1.7320508075688772, 0.9999999999999998], [1.7320508075688772, 2.], [0.8660254037844387, 2.5],
                    [0., 1.9999999999999998], [0., 0.9999999999999999], [0.8660254037844386, 0.4999999999999999]],
                   [[1.7320508075688772, -1.], [2.598076211353316, -0.5], [2.598076211353316, 0.5],
                    [1.7320508075688772, 1.], [0.8660254037844386, 0.5], [0.8660254037844386, -0.5]]]

        lines = [p_red]

        for i in range(1, 7):
            text = TexMobject(r"H_", i, "=", r"(M_b\cdot M_c)^", i, r"\cdot", "H")
            text[0].set_color(CYAN)
            text[1].set_color(CYAN)
            text[3].set_color(CYAN)
            text[6].set_color(RED)
            text.scale(0.7)
            lines.append(text)
            text.next_to(lines[i - 1], DOWN)
            text.align_to(lines[i - 1], LEFT)

            hexagon = Polygon(self.coords_to_point(coords3[i - 1][0][0], coords3[i - 1][0][1]),
                              self.coords_to_point(coords3[i - 1][1][0], coords3[i - 1][1][1]),
                              self.coords_to_point(coords3[i - 1][2][0], coords3[i - 1][2][1]),
                              self.coords_to_point(coords3[i - 1][3][0], coords3[i - 1][3][1]),
                              self.coords_to_point(coords3[i - 1][4][0], coords3[i - 1][4][1]),
                              self.coords_to_point(coords3[i - 1][5][0], coords3[i - 1][5][1]),
                              self.coords_to_point(coords3[i - 1][0][0], coords3[i - 1][0][1]), stroke_width=5,
                              fill_opacity=0.5)
            hexagon.set_color(CYAN)
            objects.append(hexagon)
            self.play(Write(text), ShowCreation(hexagon))
            self.wait()

        self.wait(10)

        self.play(Uncreate(p_red), *[Uncreate(lines[i]) for i in range(0, len(lines))])
        self.wait(3)

        props = TextMobject("Properties")
        props.set_color(YELLOW)
        props.next_to(title, DOWN)
        props.to_edge(LEFT)
        mat_a.generate_target()
        mat_a.target.next_to(props, RIGHT, buff=1.6 * RIGHT)
        mat_a.target.shift(0.25 * DOWN)
        mat_b.generate_target()
        mat_b.target.next_to(mat_a.target, DOWN)
        mat_b.target.align_to(mat_a.target, LEFT)
        mat_c.generate_target()
        mat_c.target.next_to(mat_b.target, DOWN)
        mat_c.target.align_to(mat_b.target, LEFT)

        matgroup = VGroup(mat_a.target, mat_b.target, mat_c.target)
        rect = SurroundingRectangle(matgroup)
        rect.set_color(YELLOW)

        self.add(line_a, line_b, line_c)
        self.play(Uncreate(hexagon2))
        self.play(Write(props), MoveToTarget(mat_a), MoveToTarget(mat_b), MoveToTarget(mat_c))
        self.play(GrowFromEdge(rect, UL))

        mats = ["{M_a}", "{M_b}", "{M_c}"]
        cols = [RED, MARINE, GREEN]

        lines = [props]
        for i in range(0, 3):
            text = TexMobject(mats[i], r"^2=", r"\left(\begin{array}{c c} 1 & 0\\ 0 & 1\end{array}\right)")
            text[0].set_color(cols[i])
            text[2].scale(0.7)
            text.scale(0.7)
            text.next_to(lines[i], DOWN)
            text.to_edge(LEFT)
            lines.append(text)
            self.play(Write(text))
            self.wait()

        self.wait(3)

        powers = [3, 6, 2]
        cols = [RED, GREEN, MARINE]
        for i in range(0, 3):
            text = TexMobject("(", mats[i], r"\cdot", mats[(i + 1) % 3], ")^", powers[i],
                              "=", r"\left(\begin{array}{c c} 1 & 0\\ "
                                   r"0 & 1\end{array}\right)")
            text[1].set_color(cols[i])
            text[3].set_color(cols[(i + 1) % 3])
            text[7].scale(0.7)
            text.scale(0.7)
            text.next_to(lines[i + 3], DOWN)
            text.to_edge(LEFT)
            lines.append(text)
            self.play(Write(text))
            self.wait()

        self.wait(3)

        dynkin = DynkinDiagram([3, 6], [0, 0, 1], [RED, GREEN, MARINE], ["a", "b", "c"])
        dynkin.move_to(1 * LEFT + 2 * DOWN)
        # self.play(*[Uncreate(objects[i]) for i in range(0, len(objects))])
        self.play(GrowFromCenter(dynkin))
        self.wait()

        rect2 = SurroundingRectangle(dynkin)
        rect2.set_color(YELLOW)

        self.play(GrowFromEdge(rect2, UL))
        self.wait(5)
        group = TextMobject("Group: ")
        group.scale(0.6)
        group.set_color(YELLOW)
        group.next_to(lines[len(lines) - 1], DOWN, buff=MED_LARGE_BUFF)
        group.align_to(lines[len(lines) - 1], LEFT)

        self.play(Write(group))

        group2 = TexMobject("G=<a,b,c| a^2=b^2=c^2=(ac)^2=(ab)^3=(bc)^6=1>")
        group2.scale(0.6)
        group2.next_to(group, RIGHT)
        self.play(Write(group2))
        self.wait(10)


class DynkinDiagram(VGroup):
    def __init__(self, weights, activations, colors, labels, **kwargs):
        VGroup.__init__(self, **kwargs)
        dots = []
        for i in range(0, len(activations)):
            dot = Dot()
            if i > 0:
                dot.next_to(dots[i - 1], RIGHT, buff=LARGE_BUFF)
            self.add(dot)
            dots.append(dot)
            dot.scale(2)
            if activations[i] > 0:
                dot.set_style(stroke_color=colors[i], fill_color=colors[i], fill_opacity=0.5)
            else:
                dot.set_style(stroke_color=colors[i], stroke_width=2, fill_opacity=0)
            label = TexMobject(labels[i])
            label.next_to(dot, DOWN)
            label.set_color(colors[i])
            self.add(label)
            if i > 0 and weights[i - 1] > 2:
                line = Line(dots[i - 1], dot)
                line.set_style(stroke_color=WHITE, stroke_width=4)
                if weights[i - 1] > 3:
                    weight = TexMobject(weights[i - 1])
                    weight.set_color(WHITE)
                    weight.next_to(line, UP)
                    self.add(weight)
                elif weights[i - 1] == 3:
                    weight = TexMobject(weights[i - 1])
                    weight.set_color(WHITE)
                    weight.set_style(stroke_color=WHITE, stroke_opacity=0.5, stroke_width=2, fill_opacity=0)
                    weight.next_to(line, UP)
                    self.add(weight)
                self.add(line)


class BehindTheScenes2(Scene):

    def construct(self):
        objects = []
        title = Title("Behind the scenes II")
        title.set_color(YELLOW)
        title.to_edge(UP)
        self.play(Write(title))

        dynkin = DynkinDiagram([3, 5], [0, 0, 1], [RED, GREEN, MARINE], ["a", "b", "c"])
        dynkin.next_to(title, DOWN, buff=MED_LARGE_BUFF)
        dynkin.to_edge(LEFT, buff=MED_LARGE_BUFF)
        self.play(GrowFromCenter(dynkin))
        self.wait()

        rect2 = SurroundingRectangle(dynkin)
        rect2.set_color(YELLOW)

        self.play(GrowFromEdge(rect2, UL))
        self.wait(3)

        mat_a = TexMobject("M_a =", r"\left(\begin{array}{c c c} -1 & 0 & 0\\ 0 & 1 & 0 \\ 0 & 0 & 1\end{array}\right)")
        mat_a.set_color(RED)
        mat_a.scale(0.7)
        mat_a.next_to(dynkin, DOWN, buff=LARGE_BUFF)
        mat_a.to_edge(LEFT, buff=MED_LARGE_BUFF)
        self.play(Write(mat_a))
        self.wait()

        mat_b = TexMobject("M_b =",
                           r"\frac{1}{4}\left(\begin{array}{c c c} 2 & -1-\sqrt{5} & -1+\sqrt{5}\\ -1-\sqrt{5} & 1-\sqrt{5} & 2 \\ -1+\sqrt{5} & 2 & 1+\sqrt{5}\end{array}\right)")
        mat_b.set_color(GREEN)
        mat_b[1].scale(0.5)
        mat_b[1].shift(2 * LEFT)
        mat_b.scale(0.7)
        mat_b.next_to(mat_a, DOWN)
        mat_b.to_edge(LEFT, buff=MED_LARGE_BUFF)
        self.play(Write(mat_b))
        self.wait()

        mat_c = TexMobject("M_c =", r"\left(\begin{array}{c c c} 1 & 0 & 0\\ 0 & 1 & 0 \\ 0 & 0 & -1\end{array}\right)")
        mat_c.set_color(MARINE)
        mat_c.scale(0.7)
        mat_c.next_to(mat_b, DOWN)
        mat_c.to_edge(LEFT, buff=MED_LARGE_BUFF)
        self.play(Write(mat_c))
        self.wait()

        rect2 = SurroundingRectangle(VGroup(mat_a, mat_b, mat_c))
        rect2.set_color(YELLOW)
        self.play(GrowFromEdge(rect2, UL))
        self.wait()

        props = TextMobject("Properties")
        props.set_color(YELLOW)
        props.next_to(title, DOWN)
        self.play(Write(props))
        self.wait()

        mats = ["{M_a}", "{M_b}", "{M_c}"]
        cols = [RED, MARINE, GREEN]

        lines = [props]
        for i in range(0, 3):
            text = TexMobject(mats[i], r"^2=",
                              r"\left(\begin{array}{c c c} 1 & 0 & 0\\ 0 & 1 & 0\\ 0 & 0 & 1\end{array}\right)")
            text[0].set_color(cols[i])
            text[2].scale(0.5)
            text[2].shift(0.9 * LEFT)
            text.scale(0.7)
            text.next_to(lines[i], DOWN)
            text.align_to(lines[i], LEFT)
            lines.append(text)
            self.play(Write(text))
            self.wait()

        self.wait(3)

        powers = [3, 5, 2]
        cols = [RED, GREEN, MARINE]
        for i in range(0, 3):
            text = TexMobject("(", mats[i], r"\cdot", mats[(i + 1) % 3], ")^", powers[i],
                              "=", r"\left(\begin{array}{c c c} 1 & 0 & 0\\ "
                                   r"0 & 1 & 0\\ 0 & 0 & 1\end{array}\right)")
            text[1].set_color(cols[i])
            text[3].set_color(cols[(i + 1) % 3])
            text[7].scale(0.5)
            text[7].shift(0.9 * LEFT)
            text.scale(0.7)
            text.next_to(lines[i + 3], DOWN)
            text.align_to(lines[i + 3], LEFT)
            lines.append(text)
            self.play(Write(text))
            self.wait()

        self.wait(3)

        startvector = TextMobject("Start vector")
        startvector.set_color(YELLOW)
        startvector.next_to(props, RIGHT, buff=2 * RIGHT)
        self.play(Write(startvector))
        self.wait()

        vec = TexMobject(r"\vec P=", r"\left(\begin{array}{c} 0 \\ \sqrt{5}-1 \\ \sqrt{5}+1\end{array}\right)")
        vec.set_color(YELLOW)
        vec[1].scale(0.7)
        vec[1].shift(0.5 * LEFT)
        vec.scale(0.7)
        vec.next_to(startvector, DOWN)
        vec.align_to(startvector, LEFT)
        self.play(Write(vec))
        self.wait()

        prop = TexMobject("M_a", "\cdot", r"\vec P", "=", r"\vec P")
        prop.scale(0.7)
        prop[0].set_color(RED)
        prop[2].set_color(YELLOW)
        prop[4].set_color(YELLOW)
        prop.next_to(vec, DOWN)
        prop.align_to(vec, LEFT)
        self.play(Write(prop))
        self.wait()

        prop2 = TexMobject("M_b", "\cdot", r"\vec P", "=", r"\vec P")
        prop2.scale(0.7)
        prop2[0].set_color(GREEN)
        prop2[2].set_color(YELLOW)
        prop2[4].set_color(YELLOW)
        prop2.next_to(prop, DOWN)
        prop2.align_to(prop, LEFT)
        self.play(Write(prop2))
        self.wait()

        coords = [["-2", " -2", " -2"], ["0", r"1-\sqrt{5}", r"-1-\sqrt{5}"], ["0", r"1-\sqrt{5}", r"1+\sqrt{5}"],
                  ["-2", " -2", " 2"],
                  ["-2", " 2", " -2"], ["0", r"\sqrt{5}-1", r"-1-\sqrt{5}"], ["0", r"\sqrt{5}-1", r"1+\sqrt{5}"],
                  ["-2", " 2", " 2"],
                  ["2", " -2", " -2"], [r"-1-\sqrt{5}", "0", r"1-\sqrt{5}"], [r"-1-\sqrt{5}", "0", r"\sqrt{5}-1"],
                  ["2", " -2", " 2"],
                  ["2", " 2", " -2"], [r"1-\sqrt{5}", r"-1-\sqrt{5}", "0"], [r"1-\sqrt{5}", r"1+\sqrt{5}", "0"],
                  [r"1+\sqrt{5}", "0", r"1-\sqrt{5}"],
                  ["2", " 2", " 2"], [r"\sqrt{5}-1", r"-1-\sqrt{5}", "0"], [r"\sqrt{5}-1", r"1+\sqrt{5}", "0"],
                  [r"1+\sqrt{5}", "0", r"\sqrt{5}-1"]]
        points = []
        for i in range(0, len(coords)):
            text = r"\left(\begin{array}{c}" + coords[i][0] + r"\\" + coords[i][1] + r"\\" + coords[i][
                2] + r"\end{array}\right)"
            point = TexMobject(text)
            point.scale(0.27)
            if i == 0:
                point.next_to(prop2, DOWN)
                point.align_to(prop2, LEFT)
            elif i % 4 == 0:
                point.align_to(points[i - 4], LEFT)
                point.next_to(points[i - 4], DOWN)
            else:
                point.next_to(points[i - 1], RIGHT)
            points.append(point)
            self.play(Write(point))

        rect3 = SurroundingRectangle(VGroup(*points))
        rect3.set_color(YELLOW)
        self.play(GrowFromEdge(rect3, DR))

        self.wait(10)


class BehindTheScenes3(Scene):

    def construct(self):
        objects = []
        title = Title("Behind the scenes III")
        title.set_color(YELLOW)
        title.to_edge(UP)
        self.play(Write(title))

        dynkin = DynkinDiagram([3, 7], [0, 0, 1], [RED, GREEN, MARINE], ["a", "b", "c"])
        dynkin.next_to(title, DOWN, buff=MED_LARGE_BUFF)
        dynkin.to_edge(LEFT, buff=MED_LARGE_BUFF)
        self.play(GrowFromCenter(dynkin))
        self.wait()

        rect2 = SurroundingRectangle(dynkin)
        rect2.set_color(YELLOW)

        self.play(GrowFromEdge(rect2, UL))
        self.wait(3)

        props = TextMobject("Properties")
        props.set_color(YELLOW)
        props.next_to(title, DOWN)
        props.shift(1.5 * RIGHT)

        mats = ["{M_a}", "{M_b}", "{M_c}"]
        cols = [RED, MARINE, GREEN]

        lines = [props]
        for i in range(0, 3):
            text = TexMobject(mats[i], r"^2=",
                              r"\left(\begin{array}{c c c} 1 & 0 & 0\\ 0 & 1 & 0\\ 0 & 0 & 1\end{array}\right)")
            text[0].set_color(cols[i])
            text[2].scale(0.5)
            text[2].shift(0.9 * LEFT)
            text.scale(0.7)
            text.next_to(lines[i], DOWN)
            text.align_to(lines[i], LEFT)
            lines.append(text)

        powers = [3, 7, 2]
        cols = [RED, GREEN, MARINE]
        for i in range(0, 3):
            text = TexMobject("(", mats[i], r"\cdot", mats[(i + 1) % 3], ")^", powers[i],
                              "=", r"\left(\begin{array}{c c c} 1 & 0 & 0\\ "
                                   r"0 & 1 & 0\\ 0 & 0 & 1\end{array}\right)")
            text[1].set_color(cols[i])
            text[3].set_color(cols[(i + 1) % 3])
            text[7].scale(0.5)
            text[7].shift(0.9 * LEFT)
            text.scale(0.7)
            text.next_to(lines[i + 3], DOWN)
            text.align_to(lines[i + 3], LEFT)
            lines.append(text)

        self.play(*[Write(lines[i]) for i in range(0, len(lines))])
        self.wait(3)

        mat_a = TexMobject("M_a =", r"\left(\begin{array}{c c c} -1 & 0 & 0\\ 0 & 1 & 0 \\ 0 & 0 & 1\end{array}\right)")
        mat_a.set_color(RED)
        mat_a.scale(0.7)
        mat_a.next_to(dynkin, DOWN)
        mat_a.to_edge(LEFT, buff=MED_LARGE_BUFF)

        mat_b = TexMobject("M_b =",
                           r"\frac{1}{2}\left(\begin{array}{c c c} 1 & -\sqrt{\alpha^2-3} & -\alpha\\ \sqrt{\alpha^2-3} & \alpha^2-1 & \alpha\sqrt{\alpha^2-3} \\ -\alpha & -\alpha\sqrt{\alpha^2-3} & 2-\alpha^2\end{array}\right)")
        mat_b.set_color(GREEN)
        mat_b[1].scale(0.7)
        mat_b[1].shift(1.5 * LEFT)
        mat_b.scale(0.7)
        mat_b.next_to(mat_a, DOWN)
        mat_b.to_edge(LEFT, buff=MED_LARGE_BUFF)
        mat_b2 = TexMobject(r"\alpha=2\cos\tfrac{\pi}{2}")
        mat_b2.set_color(GREEN)
        mat_b2.scale(0.5)
        mat_b2.next_to(mat_b, DOWN)
        mat_b2.to_edge(LEFT, buff=MED_LARGE_BUFF)

        mat_c = TexMobject("M_c =", r"\left(\begin{array}{c c c} 1 & 0 & 0\\ 0 & 1 & 0 \\ 0 & 0 & -1\end{array}\right)")
        mat_c.set_color(MARINE)
        mat_c.scale(0.7)
        mat_c.next_to(mat_b2, DOWN)
        mat_c.to_edge(LEFT, buff=MED_LARGE_BUFF)
        self.play(Write(mat_a), Write(mat_c))
        self.wait(3)

        self.play(Write(mat_b), Write(mat_b2))
        self.wait(3)

        rect2 = SurroundingRectangle(VGroup(mat_a, mat_b, mat_c))
        rect2.set_color(YELLOW)
        self.play(GrowFromEdge(rect2, UL))
        self.wait()

        startvector = TextMobject("Start vector")
        startvector.set_color(YELLOW)
        startvector.next_to(props, RIGHT, buff=1.5 * RIGHT)
        self.play(Write(startvector))
        self.wait()

        vec = TexMobject(r"\vec P=", r"\left(\begin{array}{c} 0 \\ \alpha \\ -\sqrt{\alpha^2-3}\end{array}\right)")
        vec.set_color(YELLOW)
        vec[1].scale(0.7)
        vec[1].shift(0.5 * LEFT)
        vec.scale(0.7)
        vec.next_to(startvector, DOWN)
        vec.align_to(startvector, LEFT)
        self.play(Write(vec))
        self.wait()

        prop = TexMobject("M_a", "\cdot", r"\vec P", "=", r"\vec P")
        prop.scale(0.7)
        prop[0].set_color(RED)
        prop[2].set_color(YELLOW)
        prop[4].set_color(YELLOW)
        prop.next_to(vec, DOWN)
        prop.align_to(vec, LEFT)
        self.play(Write(prop))
        self.wait()

        prop2 = TexMobject("M_b", "\cdot", r"\vec P", "=", r"\vec P")
        prop2.scale(0.7)
        prop2[0].set_color(GREEN)
        prop2[2].set_color(YELLOW)
        prop2[4].set_color(YELLOW)
        prop2.next_to(prop, DOWN)
        prop2.align_to(prop, LEFT)
        self.play(Write(prop2))
        self.wait()

        self.wait(10)


class HyperboloidBackground(Scene):
    def construct(self):
        hyperboloid = TexMobject("x^2-y^2+z^2=-3")
        hyperboloid.scale(3)
        hyperboloid.to_corner(DL)
        hyperboloid.set_style(stroke_opacity=1, stroke_width=4, fill_opacity=0.5)
        self.play(FadeIn(hyperboloid))
        self.wait(30)


class EverythingIsPossible(GraphScene):
    def __init__(self, **kwargs):
        GraphScene.__init__(self, x_min=-1,
                            x_max=1,
                            y_min=-1,
                            y_max=1,
                            y_axis_height=2,
                            x_axis_width=2,
                            y_axis_label="",
                            x_axis_label="",
                            axes_color=BLACK,
                            graph_origin=0 * LEFT,
                            **kwargs)

    def construct(self):
        title = Title("Every tessellation is possible")
        title.set_color(YELLOW)
        title.to_edge(UP)
        self.play(Write(title))

        self.setup_axes()

        shift_x = -4
        shift_y = 1.8
        for i in range(0, 5):
            coordinates = [self.coords_to_point(0 + shift_x, shift_y),
                           self.coords_to_point(np.cos(2 * np.pi / 6 * i) + shift_x,
                                                np.sin(2 * np.pi / 6 * i) + shift_y),
                           self.coords_to_point(np.cos(2 * np.pi / 6 * (i + 1)) + shift_x,
                                                np.sin(2 * np.pi / 6 * (i + 1)) + shift_y)]
            triangle = Polygon(*coordinates, stroke_width=5, fill_opacity=0.5)
            self.play(GrowFromCenter(triangle), run_time=0.2)

        dynkin1 = DynkinDiagram([5, 3], [0, 0, 1], [WHITE, WHITE, WHITE], ["", "", ""])
        dynkin1.shift(5.25 * LEFT)
        self.play(GrowFromCenter(dynkin1))

        shift_x = 0
        for i in range(0, 6):
            coordinates = [self.coords_to_point(0 + shift_x, shift_y),
                           self.coords_to_point(np.cos(2 * np.pi / 6 * i) + shift_x,
                                                np.sin(2 * np.pi / 6 * i) + shift_y),
                           self.coords_to_point(np.cos(2 * np.pi / 6 * (i + 1)) + shift_x,
                                                np.sin(2 * np.pi / 6 * (i + 1)) + shift_y)]
            triangle = Polygon(*coordinates, stroke_width=5, fill_opacity=0.5)
            self.play(GrowFromCenter(triangle), run_time=0.2)

        dynkin1 = DynkinDiagram([6, 3], [0, 0, 1], [WHITE, WHITE, WHITE], ["", "", ""])
        dynkin1.shift(1.25 * LEFT)
        self.play(GrowFromCenter(dynkin1))

        shift_x = 4
        for i in range(0, 7):
            coordinates = [self.coords_to_point(0 + shift_x, shift_y),
                           self.coords_to_point(np.cos(2 * np.pi / 6 * i) + shift_x,
                                                np.sin(2 * np.pi / 6 * i) + shift_y),
                           self.coords_to_point(np.cos(2 * np.pi / 6 * (i + 1)) + shift_x,
                                                np.sin(2 * np.pi / 6 * (i + 1)) + shift_y)]
            triangle = Polygon(*coordinates, stroke_width=5, fill_opacity=0.5)
            self.play(GrowFromCenter(triangle), run_time=0.2)

        dynkin1 = DynkinDiagram([7, 3], [0, 0, 1], [WHITE, WHITE, WHITE], ["", "", ""])
        dynkin1.shift(2.75 * RIGHT)
        self.play(GrowFromCenter(dynkin1))

        self.wait(10)


class Examples(Scene):
    def construct(self):
        dynkin1 = DynkinDiagram([3, 7], [1, 1, 1], [WHITE, WHITE, WHITE], ["", "", ""])
        dynkin1.shift(5 * LEFT + 3 * UP)
        self.play(FadeIn(dynkin1))
        self.wait(3)

        dynkin2 = DynkinDiagram([4, 5], [1, 1, 1], [WHITE, WHITE, WHITE], ["", "", ""])
        dynkin2.shift(3 * RIGHT + 0 * UP)
        self.play(FadeIn(dynkin2))
        self.wait(3)

        dynkin3 = DynkinDiagram([21, 3], [0, 1, 0], [WHITE, WHITE, WHITE], ["", "", ""])
        dynkin3.shift(-5 * RIGHT + 3 * DOWN)
        self.play(FadeIn(dynkin3))
        self.wait(3)

        self.wait(20)


class IntroBackground(GraphScene):
    def __init__(self, **kwargs):
        GraphScene.__init__(self, x_min=-1,
                            x_max=1,
                            y_min=-1,
                            y_max=1,
                            y_axis_height=2,
                            x_axis_width=2,
                            y_axis_label="",
                            x_axis_label="",
                            axes_color=BLACK,
                            graph_origin=0 * LEFT,
                            **kwargs)

    def construct(self):

        self.setup_axes()

        shift_x = -6.5
        shift_y = 3.8
        r3 = np.sqrt(3)
        r2 = np.sqrt(2)

        hexagons = []
        for col in range(0, 8):
            shift_y = shift_y - r3 / 4
            if col % 2 == 0:
                shift_x = shift_x + 0.75
            else:
                shift_x = shift_x - 0.75
            coordinates = []
            for i in range(0, 6):
                coordinates.append(self.coords_to_point(0.5 * np.cos(2 * np.pi / 6 * i) + shift_x,
                                                        0.5 * np.sin(2 * np.pi / 6 * i) + shift_y))

            hexagon = Polygon(*coordinates, stroke_width=5, fill_opacity=0.5)
            hexagons.append(hexagon)

        self.wait(5)
        self.play(ShowCreation(VGroup(*hexagons)), run_time=1)
        self.wait(1)

        shift_y = shift_y - 0.5
        shift_x = shift_x

        squares = []
        for col in range(0, 10):
            if col < 5:
                shift_y = shift_y - 0.5
                if col % 2 == 0:
                    shift_x = shift_x + 0.5
                else:
                    shift_x = shift_x - 0.5
            else:
                shift_x = shift_x + 0.5
                if col % 2 == 0:
                    shift_y = shift_y + 0.5
                else:
                    shift_y = shift_y - 0.5

            coordinates = []
            for i in range(0, 4):
                coordinates.append(self.coords_to_point(0.5 * np.cos(2 * np.pi / 4 * i) + shift_x,
                                                        0.5 * np.sin(2 * np.pi / 4 * i) + shift_y))

            square = Polygon(*coordinates, stroke_width=5, fill_opacity=0.5)
            squares.append(square)

        self.play(ShowCreation(VGroup(*squares)), run_time=1)
        self.wait(1)

        shift_x = shift_x + 1
        shift_y = shift_y + 0.5

        triangles = []
        for col in range(0, 10):
            if col % 2 == 0:
                shift_y = shift_y - r3 / 2
                shift_x = shift_x + 0.5
            else:
                shift_y = shift_y + r3 / 2
                shift_x = shift_x + 0.5

            for i in range(0, 2):
                coordinates = [self.coords_to_point(0 + shift_x, shift_y),
                               self.coords_to_point(np.cos(2 * np.pi / 6 * i) + shift_x,
                                                    np.sin(2 * np.pi / 6 * i) + shift_y),
                               self.coords_to_point(np.cos(2 * np.pi / 6 * (i + 1)) + shift_x,
                                                    np.sin(2 * np.pi / 6 * (i + 1)) + shift_y)]
                triangle = Polygon(*coordinates, stroke_width=5, fill_opacity=0.5)
                triangles.append(triangle)

        v_group = VGroup(*triangles)
        v_group.rotate(np.pi / 2)
        v_group.shift(self.coords_to_point(5, 3.5))
        self.play(ShowCreation(v_group), run_time=1)
        self.wait(4)

        coordinates = []
        for i in range(0, 5):
            coordinates.append(self.coords_to_point(0.7 * np.cos(2 * np.pi / 5 * i) + shift_x,
                                                    0.7 * np.sin(2 * np.pi / 5 * i) + shift_y))

        heptagon = Polygon(*coordinates, stroke_width=5, fill_opacity=0.5, stroke_color=RED, fill_color=RED)
        heptagon.move_to(self.coords_to_point(5.3, -3))
        self.play(GrowFromCenter(heptagon))

        coordinates = []
        for i in range(0, 7):
            coordinates.append(self.coords_to_point(0.5 * np.cos(2 * np.pi / 7 * i) + shift_x,
                                                    0.5 * np.sin(2 * np.pi / 7 * i) + shift_y))

        nonagon = Polygon(*coordinates, stroke_width=5, fill_opacity=0.5, stroke_color=GREEN, fill_color=GREEN)
        nonagon.move_to(self.coords_to_point(3.7, -3))
        self.play(GrowFromCenter(nonagon))

        coordinates = []
        for i in range(0, 9):
            coordinates.append(self.coords_to_point(0.3 * np.cos(2 * np.pi / 9 * i) + shift_x,
                                                    0.3 * np.sin(2 * np.pi / 9 * i) + shift_y))

        nonagon = Polygon(*coordinates, stroke_width=5, fill_opacity=0.5, stroke_color=YELLOW, fill_color=YELLOW)
        nonagon.move_to(self.coords_to_point(6.7, -3))
        self.play(GrowFromCenter(nonagon))

        self.wait(10)


class Explanation(Scene):
    def construct(self):
        title = Title("The calculation of the matrices")
        title.set_color(YELLOW)
        title.to_edge(UP)
        self.play(Write(title))

        dynkin1 = DynkinDiagram([5, 3], [0, 0, 1], [RED, GREEN, BLUE], ["a", "b", "c"])
        dynkin1.next_to(title, DOWN)
        dynkin1.shift(5.25 * LEFT)
        self.play(GrowFromCenter(dynkin1))

        dynkin1 = DynkinDiagram([6, 3], [0, 0, 1], [RED, GREEN, BLUE], ["a", "b", "c"])
        dynkin1.next_to(title, DOWN)
        self.play(GrowFromCenter(dynkin1))

        dynkin1 = DynkinDiagram([7, 3], [0, 0, 1], [RED, GREEN, BLUE], ["a", "b", "c"])
        dynkin1.next_to(title, DOWN)
        dynkin1.shift(5.25 * RIGHT)
        self.play(GrowFromCenter(dynkin1))
        self.wait(3)

        rel1 = TexMobject(r"{\vec n_a}", r"\cdot", r"{\vec n_c}", "=0")
        rel1[0].set_color(RED)
        rel1[2].set_color(BLUE)
        rel1.shift(2 * DOWN + 0.5 * LEFT)
        rel2 = TexMobject(r" {\vec n_b}", r"\cdot", r"{\vec n_c}", r"=\cos\tfrac{\pi}{3}")
        rel2[0].set_color(GREEN)
        rel2[2].set_color(BLUE)
        rel2.next_to(rel1, DOWN)
        rel2.align_to(rel1, LEFT)

        rel3 = TexMobject(r" {\vec n_a}", r"\cdot", r"{\vec n_b}", r"=\cos\tfrac{\pi}{6}")
        rel3[0].set_color(RED)
        rel3[2].set_color(GREEN)
        rel3.next_to(rel2, DOWN)
        rel3.align_to(rel2, LEFT)

        rel4 = TexMobject(r" {\vec n_a}", r"\cdot", r"{\vec n_b}", r"=\cos\tfrac{\pi}{5}")
        rel4[0].set_color(RED)
        rel4[2].set_color(GREEN)
        rel4.next_to(rel3, LEFT)
        rel4.to_edge(LEFT)

        rel5 = TexMobject(r" {\vec n_a}", r"\cdot", r"{\vec n_b}", r"=\cos\tfrac{\pi}{7}")
        rel5[0].set_color(RED)
        rel5[2].set_color(GREEN)
        rel5.next_to(rel3, RIGHT)
        rel5.to_edge(RIGHT)

        self.play(Write(rel1))
        self.wait()
        self.play(Write(rel2))
        self.wait()
        self.play(Write(rel3), Write(rel4), Write(rel5))
        self.play()
        self.wait(3)

        eqs = VGroup(rel1, rel2, rel3, rel4, rel5)
        self.play(eqs.shift, 2.5 * UP)
        self.wait(3)

        ansatz = TexMobject(r" {\vec n_a}", "=", r"\left(\begin{array}{c} 1\\0\\0\end{array}\right)", ", ",
                            r" {\vec n_c}", "=", r"\left(\begin{array}{c} 0\\0\\1\end{array}\right)")
        ansatz2 = TexMobject(",", r" {\vec n_b}", "=", r"\left(\begin{array}{c} n_x\\n_y\\n_z\end{array}\right)")
        solz = TexMobject(r"n_z", r"=\cos\tfrac{\pi}{3}=\tfrac{1}{2}")
        solx2 = TexMobject(r"n_x", r"=\cos\tfrac{\pi}{6}=\tfrac{\sqrt{3}}{2}")
        solx1 = TexMobject(r"n_x", r"=\cos\tfrac{\pi}{5}=\tfrac{1}{4}(1+\sqrt{5})")
        solx3 = TexMobject(r"n_x", r"=\cos\tfrac{\pi}{7}")
        norm1 = TexMobject("n_x^2+","{n_y}","^2+", "{n_z}", "^2=1")
        norm2 = TexMobject("n_x^2+","{n_y}","^2+", "{n_z}", "^2=1")
        norm3 = TexMobject("n^2_x", "-", "{n_y}","^2+", "{n_z}", "^2=1")
        solz[0].set_color(YELLOW)
        solx2[0].set_color(YELLOW)
        solx1[0].set_color(YELLOW)
        solx3[0].set_color(YELLOW)
        norm1[1].set_color(YELLOW)
        norm2[1].set_color(YELLOW)
        norm3[1].set_color(RED)
        norm3[2].set_color(YELLOW)
        ansatz[0].set_color(RED)
        ansatz[4].set_color(BLUE)
        ansatz2[1].set_color(GREEN)
        ansatz.scale(0.7)
        ansatz2.scale(0.7)
        solz.scale(0.7)
        solx1.scale(0.7)
        solx2.scale(0.7)
        solx3.scale(0.7)
        norm1.scale(0.7)
        norm2.scale(0.7)
        norm3.scale(0.7)
        eqs2 = VGroup(eqs, ansatz)
        ansatz.next_to(eqs, DOWN)
        ansatz.shift(LEFT)
        self.play(Uncreate(rel1), Write(ansatz))
        self.play(eqs2.shift, 0.5 * UP)
        solz.next_to(eqs2, DOWN)
        ansatz2.next_to(ansatz, RIGHT)
        self.play(Write(ansatz2))
        self.wait()
        self.play(Write(solz), Uncreate(rel2))
        eqs3 = VGroup(eqs2, ansatz2, solz)
        self.play(eqs3.shift, UP)
        self.wait()
        solx1.next_to(eqs3, DOWN)
        solx2.next_to(eqs3, DOWN)
        solx3.next_to(eqs3, DOWN)
        solx1.to_edge(LEFT)
        solx3.to_edge(RIGHT)
        self.play(Write(solx1), Uncreate(rel4))
        self.play(Write(solx2), Uncreate(rel3))
        self.play(Write(solx3), Uncreate(rel5))
        eqs4 = VGroup(eqs3, solx1, solx2, solx3)
        self.wait()
        self.play(eqs4.shift, UP)
        self.wait()
        norm2.next_to(eqs4, DOWN)
        norm1.next_to(eqs4, DOWN)
        norm3.next_to(eqs4, DOWN)
        norm1.to_edge(LEFT)
        norm3.to_edge(RIGHT)
        self.play(Write(norm1), Write(norm2))
        self.wait()
        self.play(Write(norm3))
        self.wait()
        eqs5 = VGroup(eqs4, norm1, norm2, norm3)

        n1 = TexMobject(r"{\vec n}_b", r"=\left(\begin{array}{c} \tfrac{1}{4}(\sqrt{5}+1)\\\tfrac{1}{4}(\sqrt{"
                                       r"5}-1)\\\tfrac{1}{2}\end{array}\right)")
        n2 = TexMobject(r"{\vec n}_b",
                        r"=\left(\begin{array}{c} \tfrac{\sqrt{3}}{2}\\0\\\tfrac{1}{2}\end{array}\right)")
        n3 = TexMobject(r"{\vec n}_b", r"=\left(\begin{array}{c} \cos\tfrac{\pi}{7}\\\tfrac{1}{2}\sqrt{4\cos^2\tfrac{"
                                       r"\pi}{7}-3}\\\tfrac{1}{2}\end{array}\right)")
        n1[0].set_color(YELLOW)
        n2[0].set_color(YELLOW)
        n3[0].set_color(YELLOW)
        n1.scale(0.5)
        n2.scale(0.5)
        n3.scale(0.5)
        n1.next_to(norm1, DOWN)
        n2.next_to(norm2, DOWN)
        n3.next_to(norm3, DOWN)
        n1.to_edge(LEFT)
        n3.to_edge(RIGHT)

        self.play(Write(n1))
        self.play(Write(n2))
        self.play(Write(n3))
        self.wait(3)

        self.play(Uncreate(eqs5))
        eqs6 = VGroup(n1, n2, n3)
        self.play(eqs6.shift, 3.5 * UP)
        self.wait(3)

        point = TexMobject(r"\vec P", "=", r"\left(\begin{array}{c} x\\y\\z\end{array}\right)")
        point.set_color(YELLOW)
        point[0].scale(0.7)
        point[1].scale(0.7)
        point[2].scale(0.5)
        point[2].shift(0.5 * LEFT)
        point.next_to(n2, DOWN)
        point.shift(3 * LEFT)

        dist = TexMobject("d", "=", r"\vec n\cdot \vec P")
        dist.set_color(YELLOW)
        dist.scale(0.7)
        dist.next_to(point, DOWN)
        dist.shift(-0.2 * LEFT)

        self.play(Write(point))
        self.wait()
        self.play(Write(dist))

        pprime = TexMobject(r"\vec P'", "=", r"\vec P-2 (\vec n\cdot \vec P) \vec n")
        pprime.set_color(YELLOW)
        pprime.scale(0.7)
        pprime.next_to(dist, DOWN)
        pprime.shift(0.7 * RIGHT)

        self.wait()
        self.play(Write(pprime))
        self.wait()

        pprime2 = TexMobject(r"\vec P'", "=", r" M \cdot \vec P")
        pprime2.set_color(WHITE)
        pprime2.scale(0.7)
        pprime2.next_to(pprime, DOWN)
        pprime2.shift(0.7 * LEFT)

        self.play(Write(pprime2))
        self.wait()

        matrix = TexMobject(r"M_{ij}", "=", r"\delta_{ij}-2 n_i n_j")
        matrix.set_color(WHITE)
        matrix.scale(0.7)
        matrix.next_to(pprime2, DOWN)
        matrix.shift(0.3 * RIGHT)

        self.play(Write(matrix))
        self.wait()

        group = VGroup(point, dist, pprime, pprime2, matrix)
        group.generate_target()
        group.target.to_edge(LEFT)
        group.target.shift(0.2 * LEFT)
        surround = SurroundingRectangle(group.target)
        surround.set_color(YELLOW)

        self.play(MoveToTarget(group), GrowFromEdge(surround, LEFT))
        self.wait()

        matrix2 = TexMobject("M=",
                             r"-2\left(\begin{array}{c  c   c} n_x^2-\tfrac{1}{2} & n_x n_y & n_x n_z\\ n_y n_x & n_y^2-\tfrac{1}{2} & n_y n_z\\ n_z n_x & n_z n_y & n_z^2-\tfrac{1}{2}\end{array}\right)")
        matrix2.scale(0.475)
        matrix2.next_to(group, RIGHT, buff=LARGE_BUFF)
        matrix2.shift(UP-0.4*RIGHT)

        self.play(Write(matrix2))

        plugin = TexMobject(r"n_x=\tfrac{\sqrt{3}}{2}", ", ", "n_y=0", ",", r"n_z=\tfrac{1}{2}")
        plugin.scale(0.5)
        plugin.next_to(matrix2, DOWN)
        self.play(Write(plugin))
        self.wait()

        matrix2b = TexMobject("M_b=",
                              r"\left(\begin{array}{c c c} -\tfrac{1}{2} & 0 & -\tfrac{\sqrt{3}}{2}\\0 & 1 & 0 \\ -\tfrac{\sqrt{3}}{2} & 0  & \tfrac{1}{2}\end{array}\right)")
        matrix2b.scale(0.5)
        matrix2b.set_color(GREEN)
        matrix2b.next_to(plugin, DOWN)

        self.play(Write(matrix2b))
        self.wait()

        matrix3 = TexMobject("M=",
                             r"-2\left(\begin{array}{c  c   c} n_x^2-\tfrac{1}{2} & -n_x n_y & n_x n_z\\ n_y n_x & -n^2_y-\tfrac{1}{2} & n_y n_z\\ n_z n_x & - n_z n_y & n_z^2-\tfrac{1}{2}\end{array}\right)")
        matrix3.scale(0.475)
        matrix3.next_to(matrix2, RIGHT, buff=MED_LARGE_BUFF)

        self.play(Write(matrix3))
        self.wait()

        self.wait(10)
