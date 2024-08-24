#!/usr/bin/env python

from manimlib import *
import operator as op


class Intro(Scene):
    def construct(self):
        title = Title("Modelling a soccerball ", tex_to_color_map={"soccerball": YELLOW})
        title.move_to(LEFT * 3)
        title.to_edge(UP)
        h_line = Line(LEFT, RIGHT).scale(3)
        h_line.next_to(title, DOWN, buff=0)

        self.play(Write(title),
                  Write(h_line))

        items = BulletedList(
            "Free Fall Motion",
            "Elastic Collisions",
            "Simple Harmonic Motion",
            "Damped Harmonic Motion"
        )

        colors = [RED, YELLOW, ORANGE, BLUE]

        for (item, color) in zip(items, colors):
            item.set_color(color)

        items.scale(0.8)
        items.to_edge(LEFT, buff=LARGE_BUFF)

        rect = ScreenRectangle()
        rect.set_width(FRAME_WIDTH - items.get_width() - 2)
        rect.next_to(items, RIGHT, MED_LARGE_BUFF)

        for item in items:
            self.play(FadeIn(item))
            self.wait(3)

        self.wait(6)


class FreeFall(Scene):
    def __init__(self, **kwargs):
        Scene.__init__(self, x_min=0,
                       x_max=7,
                       y_min=0,
                       y_max=4,
                       y_axis_height=4,
                       x_axis_width=7,
                       y_axis_label="",
                       x_axis_label="t",
                       graph_origin=DOWN * 3 + LEFT * 2,
                       **kwargs)

    def fall(self, t):
        if t < 2:
            return 4 - t ** 2
        elif t < 6:
            return 4 * (t - 2) - (t - 2) ** 2
        else:
            return 4 * (t - 6) - (t - 6) ** 2

    def construct(self):
        title = Title("Free Fall")
        title.set_color(RED)
        title.to_edge(UP)
        h_line = Line(LEFT, RIGHT).scale(3)
        h_line.next_to(title, DOWN, buff=SMALL_BUFF)

        forcelaw = Tex(r"\vec F =  m\cdot \vec g")
        forcelaw.set_color(YELLOW)
        forcelaw.stre

        motion = Tex(r"\vec r=\vec r_0 + \vec v_0\cdot t+\tfrac{1}{2}\cdot \vec{g}\cdot t^2")
        motion.set_color(BLUE)
        motion.move_to(RIGHT * 3 + UP * 2)

        forcelaw.move_to(LEFT * 3 + UP * 2)

        self.play(
            Write(title),
        )

        self.wait(3)

        self.play(
            Write(forcelaw)
        )
        self.play(
            Write(motion)
        )

        self.setup_axes()
        self.show_images()
        self.wait(3)
        self.play(FadeOut(self.axes))
        self.play(ApplyMethod(forcelaw.to_edge, LEFT, buff=LARGE_BUFF))
        self.play(ApplyMethod(motion.next_to, forcelaw, DOWN, buff=SMALL_BUFF))
        self.play(ApplyMethod(motion.to_edge, LEFT, buff=LARGE_BUFF))
        self.wait(30)

    def show_images(self):

        image = self.get_physics_image()

        anims = []

        graph = self.get_graph(self.fall, color=BLUE, x_min=0, x_max=7)
        anims.append(ShowCreation(graph, run_time=14, rate_func=linear))

        if hasattr(image, "fade_in_anim"):
            anims.append(image.fade_in_anim)
        else:
            anims.append(FadeIn(image))

        if hasattr(image, "continual_animations"):
            self.add(*image.continual_animations)

        self.play(*anims)
        self.play(FadeOut(graph), image.fade_out_anim)

    def get_physics_image(self):

        ball = Circle(radius=0.2)
        ball.move_to(LEFT * 3 + UP * 0)
        ball.set_stroke(RED, 2)
        ball.set_fill(opacity=1, color=RED)

        weight = Square()
        weight.set_stroke(width=2)
        weight.set_color(RED)
        weight.set_fill(opacity=1, color=GREY)
        weight.set_height(1)
        weight.stretch(2, 0)
        weight.move_to(LEFT * 3 + DOWN * 3.5)

        point = Circle(radius=0.1)
        point.set_color(RED)
        point.set_fill(opacity=1, color=GREY)

        t_tracker = ValueTracker(0)

        group = VGroup(ball, weight, point)

        group.continual_animations = [
            t_tracker.add_updater(
                lambda tracker, dt: tracker.set_value(
                    tracker.get_value() + dt / 2
                )
            ),
            ball.add_updater(
                lambda s: s.move_to(UP * (self.fall(t_tracker.get_value()) - 3) + LEFT * 3)
            ),
            point.add_updater(
                lambda p: p.move_to(
                    self.coords_to_point(t_tracker.get_value(), self.fall(t_tracker.get_value())))
            )
        ]

        def update_group_style(alpha):
            ball.set_stroke(width=2 * alpha)
            ball.set_fill(opacity=alpha)
            weight.set_fill(opacity=alpha)
            weight.set_stroke(width=2 * alpha)
            point.set_fill(opacity=alpha)
            point.set_stroke(width=2 * alpha)

        group.fade_in_anim = UpdateFromAlphaFunc(
            group,
            lambda g, a: update_group_style(a)
        )
        group.fade_out_anim = UpdateFromAlphaFunc(
            group,
            lambda g, a: update_group_style(1 - a)
        )

        return group


class ElasticCollision(Scene):
    def __init__(self, **kwargs):
        Scene.__init__(self, x_min=0,
                       x_max=14,
                       y_min=0,
                       y_max=8,
                       y_axis_height=8,
                       x_axis_width=14,
                       y_axis_label="",
                       x_axis_label="",
                       graph_origin=DOWN * 4 + LEFT * 7,
                       axes_color=BLACK,
                       **kwargs)

    def construct(self):
        title = Title("Elastic Collision")
        title.set_color(RED)
        title.to_edge(UP)
        h_line = Line(LEFT, RIGHT).scale(3)
        h_line.next_to(title, DOWN, buff=SMALL_BUFF)

        energy = Tex(r"\tfrac{1}{2}m_1 \cdot \vec v_1^2 + \tfrac{1}{2}m_2 \cdot \vec v_2^2 ="
                     r" \tfrac{1}{2}m_1 \cdot \vec u_1^2 + \tfrac{1}{2}m_2 \cdot \vec u_2^2 ")

        energy.set_color(YELLOW)

        momentum = Tex(r"m_1 \cdot \vec v_1 + m_2 \cdot \vec v_2 = m_1 \cdot \vec u_1 + m_2 \cdot \vec u_2")
        momentum.set_color(BLUE)
        momentum.move_to(RIGHT * 3 + UP * 1)

        energy.move_to(LEFT * 1 + UP * 2)

        self.play(
            Write(title),
        )

        self.wait(5)

        self.play(
            Write(energy)
        )
        self.play(
            Write(momentum)
        )

        self.wait(5)

        grid = NumberPlane()
        # self.add(grid)
        self.setup_axes()
        graphs = self.show_images()
        image = self.show_result()
        self.show_equations()
        self.wait(10)
        self.play(image.fade_out_anim)
        self.play(FadeOut(graphs[0]), FadeOut(graphs[1]))

        self.play(
            ApplyMethod(energy.scale, 0.7),
            ApplyMethod(momentum.scale, 0.8)
        )
        self.play(
            ApplyMethod(energy.move_to, RIGHT * (0.5 + energy.get_width() / 2) + UP * 2),
            ApplyMethod(momentum.move_to, RIGHT * (0.5 + momentum.get_width() / 2) + UP * 1)
        )

        self.wait(100)

    def fall(self, t):
        if t < 2:
            return 4 - t ** 2
        elif t < 6:
            return 4 * (t - 2) - (t - 2) ** 2
        else:
            return 4 * (t - 6) - (t - 6) ** 2

    # the calculations for the animation is contained in Collision.nb (worbaseVideo)
    def motion2(self, t):
        if t < 6.57574:
            return t - 7, -1.424264
        else:
            return -0.42426 + 0.765685 * (t - 6.57574), -1.424264 - 0.565685 * (t - 6.57574)

    def motion1(self, t):
        if t < 6.57574:
            return 0, -1
        else:
            return 0.0595786 * (t - 6.57574), -1 + 0.141421 * (t - 6.57574)

    def graph2(self, t):
        if t < 6.57574:
            return 2.57574
        else:
            return 2.57574 - 0.738796 * (t - 6.57574)

    def graph1(self, t):
        if t < 7:
            return 3
        else:
            return 3 + 2.41421 * (t - 7)

    def show_images(self):
        image = self.get_physics_image()
        anims = []
        graph2 = self.get_graph(self.graph2, color=RED, x_min=0, x_max=28)
        graph1 = self.get_graph(self.graph1, color=GREEN, x_min=7, x_max=28)

        fr = 6.57574 / 28
        fr1 = 0.25

        anims.append(ShowCreation(graph2, run_time=28, rate_func=lambda x: x if x < fr else fr + 0.765685 * (x - fr)))
        anims.append(
            ShowCreation(graph1, run_time=28, rate_func=lambda x: 0 if x < fr1 else 0.0585786 / 0.7 * (x - fr1)))

        if hasattr(image, "fade_in_anim"):
            anims.append(image.fade_in_anim)
        else:
            anims.append(FadeIn(image))

        if hasattr(image, "continual_animations"):
            self.add(*image.continual_animations)

        self.play(*anims)
        self.play(image.fade_out_anim)
        return graph1, graph2

    def show_result(self):
        image = self.get_physics_image2()
        anims = []

        if hasattr(image, "fade_in_anim"):
            anims.append(image.fade_in_anim)
        else:
            anims.append(FadeIn(image))

        if hasattr(image, "continual_animations"):
            self.add(*image.continual_animations)

        self.play(*anims)
        return image

    def show_equations(self):
        image = self.get_equations()
        anims = []

        if hasattr(image, "fade_in_anim"):
            anims.append(image.fade_in_anim)
        else:
            anims.append(FadeIn(image))

        if hasattr(image, "continual_animations"):
            self.add(*image.continual_animations)
        self.play(*anims)

    def get_physics_image(self):

        ball = Circle(radius=0.4)
        ball.move_to(LEFT * 0 + UP * 0)
        ball.set_stroke(RED, 2)
        ball.set_fill(opacity=1, color=GREEN)

        ball2 = Circle(radius=0.2)
        ball2.move_to(LEFT * 3 + DOWN * 0.5)
        ball2.set_stroke(GREEN, 2)
        ball2.set_fill(opacity=1, color=YELLOW)

        t_tracker = ValueTracker(0)

        group = VGroup(ball, ball2)

        group.continual_animations = [
            t_tracker.add_updater(
                lambda tracker, dt: tracker.set_value(
                    tracker.get_value() + dt
                )
            ),
            ball2.add_updater(
                lambda s: s.move_to(RIGHT * self.motion2(t_tracker.get_value())[0] +
                                    UP * self.motion2(t_tracker.get_value())[1])
            ),
            ball.add_updater(
                lambda s: s.move_to(RIGHT * self.motion1(t_tracker.get_value())[0] +
                                    UP * self.motion1(t_tracker.get_value())[1])
            ),

        ]

        def update_group_style(alpha):
            ball.set_stroke(width=2 * alpha)
            ball.set_fill(opacity=alpha)

            ball2.set_stroke(width=2 * alpha)
            ball2.set_fill(opacity=alpha)

        group.fade_in_anim = UpdateFromAlphaFunc(
            group,
            lambda g, a: update_group_style(a)
        )
        group.fade_out_anim = UpdateFromAlphaFunc(
            group,
            lambda g, a: update_group_style(1 - a)
        )

        return group

    def get_physics_image2(self):

        ball = Circle(radius=0.4)
        ball.move_to(LEFT * 0 + UP * (-1))
        ball.set_stroke(RED, 2)
        ball.set_fill(opacity=1, color=GREEN)

        ball2 = Circle(radius=0.2)
        ball2.move_to(DOWN * 1.42426 + LEFT * 0.42426)
        ball2.set_stroke(GREEN, 2)
        ball2.set_fill(opacity=1, color=YELLOW)

        line = Line(LEFT, RIGHT).scale(2)
        line.shift(DOWN * 1 + LEFT * 2)
        line.set_stroke(YELLOW, 2)

        arrow = DoubleArrow()
        arrow.set_stroke(YELLOW, 1)
        arrow.set_fill(YELLOW)
        arrow.scale(0.42426 / 2)
        arrow.shift(DOWN * (1 + 0.42426 / 2))
        arrow.shift(LEFT * 3)
        arrow.rotate(90 * DEGREES)

        b = Tex("b")
        b.set_color(YELLOW)
        b.next_to(arrow, LEFT, buff=SMALL_BUFF)

        group = VGroup(ball, ball2, line, arrow, b)

        def update_group_style(alpha):
            ball.set_stroke(width=2 * alpha)
            ball.set_fill(opacity=alpha)
            ball2.set_stroke(width=2 * alpha)
            ball2.set_fill(opacity=alpha)
            line.set_stroke(width=2 * alpha)
            arrow.set_stroke(width=1 * alpha)
            arrow.set_fill(opacity=alpha)
            b.set_fill(opacity=alpha)

        group.fade_in_anim = UpdateFromAlphaFunc(
            group,
            lambda g, a: update_group_style(a)
        )
        group.fade_out_anim = UpdateFromAlphaFunc(
            group,
            lambda g, a: update_group_style(1 - a)
        )
        return group

    def get_equations(self):
        eq1 = Tex(
            r"\vec u_1 = \left(\begin{array}{c} 1-\cos\theta\\ \sin\theta\end{array}\right) \tfrac{m_2}{M} v_2 ")
        eq2 = Tex(
            r"\vec u_2 = \left(\begin{array}{c} \tfrac{m_2}{m_1}+\cos\theta\\ -\sin\theta\end{array}\right) \tfrac{m_1}{M} v_2 ")
        eq3 = Tex(r"M=m_1+m_2")
        eq4 = Tex(r"\cos\tfrac{\theta}{2}=\frac{b}{r_1+r_2}")
        eq1.scale(0.8)
        eq2.scale(0.8)
        eq3.scale(0.6)
        eq4.scale(0.6)
        eq1.move_to(DOWN + RIGHT * (2 + eq1.get_width() / 2))
        eq2.move_to(DOWN * 2 + RIGHT * (2 + eq2.get_width() / 2))
        eq3.move_to(DOWN * 3 + RIGHT * (3 + eq3.get_width() / 2))
        eq4.move_to(DOWN * 3.5 + RIGHT * (3 + eq4.get_width() / 2))

        group = VGroup(eq1, eq2, eq3, eq4)

        def update_group_style(alpha):
            eq1.set_fill(opacity=alpha)
            eq2.set_fill(opacity=alpha)
            eq3.set_fill(opacity=alpha)
            eq4.set_fill(opacity=alpha)

        group.fade_in_anim = UpdateFromAlphaFunc(
            group,
            lambda g, a: update_group_style(a)
        )
        group.fade_out_anim = UpdateFromAlphaFunc(
            group,
            lambda g, a: update_group_style(1 - a)
        )

        return group


class SimpleHarmonicMotion(Scene):
    def __init__(self, **kwargs):
        Scene.__init__(self, x_min=0,
                       x_max=7,
                       y_min=-1,
                       y_max=1,
                       y_axis_height=3,
                       x_axis_width=7,
                       y_axis_label="",
                       x_axis_label="t",
                       graph_origin=DOWN * 1.5 + LEFT * 2,
                       **kwargs)

    image_width = 2.5
    image_height = 3.5
    origin = DOWN * 0 + RIGHT * 0
    run_time = 10
    frequency = 1

    def oscillation(self, x):
        return -np.cos(TAU * self.frequency * x)

    def construct(self):
        title = Title("Simple Harmonic Motion")
        title.set_color(ORANGE)
        title.to_edge(UP)
        h_line = Line(LEFT, RIGHT).scale(3)
        h_line.next_to(title, DOWN, buff=SMALL_BUFF)

        forcelaw = Tex(r"\vec F = - k \vec r")
        forcelaw.set_color(YELLOW)

        motion = Tex(r"\vec r=-\vec r_0\cdot \cos(\omega\cdot t) \hspace{3em} \omega^2=\frac{k}{m}")
        motion.set_color(BLUE)
        motion.move_to(RIGHT * 3 + UP * 2)

        forcelaw.move_to(LEFT * 3 + UP * 2)

        self.play(
            Write(title),
            # ShowCreation(h_line)
        )

        self.wait(3)

        self.play(
            Write(forcelaw)
        )
        self.play(
            Write(motion)
        )

        grid = NumberPlane()
        # self.add(grid)
        self.setup_axes()
        self.show_images(1, 1)
        self.show_images(4, 1)
        self.show_images(4, 1 / 4)

    def show_images(self, spring_constant, mass):
        self.frequency = 1 / 2 * np.sqrt(spring_constant / mass)

        image = self.get_physics_image(spring_constant, mass)

        anims = []

        graph = self.get_graph(self.oscillation, color=BLUE, x_min=0, x_max=7)
        anims.append(ShowCreation(graph, run_time=14, rate_func=linear))

        if hasattr(image, "fade_in_anim"):
            anims.append(image.fade_in_anim)
        else:
            anims.append(FadeIn(image))

        if hasattr(image, "continual_animations"):
            self.add(*image.continual_animations)

        self.play(*anims)
        self.play(FadeOut(graph), image.fade_out_anim)

    def get_physics_image(self, spring_constant, mass):
        t_max = 6.5  # number of windings
        r = 0.4  # overlap of the windings //simulated view angle

        spring = ParametricCurve(
            lambda t: op.add(
                r * (np.sin(TAU * t) * RIGHT + (np.cos(TAU * t) - 1) * UP),
                t * DOWN,
            ),
            t_min=0, t_max=t_max,
            color=WHITE,
            stroke_width=2
        )
        spring.move_to(LEFT * 3 + DOWN * 6.5)
        spring.set_stroke(YELLOW, 2 * spring_constant)
        spring.stretch(2, 1)

        weight = Square()
        weight.set_stroke(width=2)
        weight.set_color(RED)
        weight.set_fill(opacity=1, color=GREY)
        weight.set_height(0.5)
        weight.stretch(2 * mass, 0)

        point = Circle(radius=0.1)
        point.set_color(RED)
        point.set_fill(opacity=1, color=GREY)
        point.move_to(self.origin)

        line = Line(weight.get_center(), point.get_center())

        t_tracker = ValueTracker(0)

        group = VGroup(spring, weight, point)

        group.continual_animations = [
            t_tracker.add_updater(
                lambda tracker, dt: tracker.set_value(
                    tracker.get_value() + dt / 2
                )
            ),
            spring.add_updater(
                lambda s: s.stretch_to_fit_height(
                    2 + 1.5 * np.cos(TAU * self.frequency * t_tracker.get_value()),
                    about_edge=UP
                )
            ),
            weight.add_updater(
                lambda w: w.move_to(spring.get_bottom() + DOWN * weight.get_height() / 2)
            ),
            point.add_updater(
                lambda p: p.move_to(
                    self.coords_to_point(t_tracker.get_value(), self.oscillation(t_tracker.get_value())))
            )
        ]

        def update_group_style(alpha):
            spring.set_stroke(width=spring_constant * alpha)
            weight.set_fill(opacity=alpha)
            weight.set_stroke(width=2 * alpha)
            point.set_fill(opacity=alpha)
            point.set_stroke(width=2 * alpha)

        group.fade_in_anim = UpdateFromAlphaFunc(
            group,
            lambda g, a: update_group_style(a)
        )
        group.fade_out_anim = UpdateFromAlphaFunc(
            group,
            lambda g, a: update_group_style(1 - a)
        )

        return group


def damped_oscillations2(x):
    return -np.exp(-0.2 * x)


def damped_oscillations3(x):
    return -(1 + TAU * x) * np.exp(-TAU * x)


def damped_oscillations1(x):
    return -np.cos(TAU * x) * np.exp(-0.1 * x)


class DampedHarmonicMotion(Scene):
    def __init__(self, **kwargs):
        Scene.__init__(self, x_min=0,
                       x_max=7,
                       y_min=-1,
                       y_max=1,
                       y_axis_height=3,
                       x_axis_width=7,
                       y_axis_label="",
                       x_axis_label="t",
                       graph_origin=DOWN * 1.5 + LEFT * 2,
                       **kwargs)

    image_width = 2.5
    image_height = 3.5
    origin = DOWN * 0 + RIGHT * 0
    run_time = 10
    frequency = 1

    def oscillation(self, x):
        return -np.cos(TAU * self.frequency * x)

    def construct(self):
        title = Title("Damped Harmonic Motion")
        title.set_color(BLUE)
        title.to_edge(UP)
        h_line = Line(LEFT, RIGHT).scale(3)
        h_line.next_to(title, DOWN, buff=SMALL_BUFF)

        forcelaw = Tex(r"\vec F = - k\cdot \vec r-b\cdot \vec v")
        forcelaw.set_color(YELLOW)

        motion = Tex(
            r"\vec r=-\vec r_0\cdot {\rm e}^{-\tfrac{b}{2m}t} \cos(\omega\cdot t) \hspace{2em} \omega^2=\frac{k}{m}-\frac{b^2}{4m^2}")
        motion.set_color(BLUE)
        motion.scale(0.75)
        motion.move_to(RIGHT * 3 + UP * 1)

        forcelaw.move_to(LEFT * 3 + UP * 2)

        self.play(
            Write(title),
            # ShowCreation(h_line)
        )

        self.wait(3)

        self.play(
            Write(forcelaw)
        )
        self.play(
            Write(motion)
        )

        items = BulletedList(
            "Underdamping",
            "Overdamping",
            "Critical"
        )
        items.scale(0.8)

        colors = [WHITE, WHITE, WHITE]

        for (item, color) in zip(items, colors):
            item.set_color(color)

        items.to_edge(LEFT, buff=SMALL_BUFF)

        grid = NumberPlane()
        # self.add(grid)
        self.setup_axes()
        self.play(Write(items[0]))
        self.show_images(1)
        self.play(FadeOut(items[0]))
        self.play(Write(items[1]))
        self.show_images(-1)
        self.play(FadeOut(items[1]))
        self.play(Write(items[2]))
        self.show_images(0)
        self.play(FadeOut(items[2]))

    def show_images(self, case):
        image = self.get_physics_image(case)
        anims = []

        if case == 1:
            graph = self.get_graph(damped_oscillations1, color=BLUE, x_min=0, x_max=7)
        elif case == -1:
            graph = self.get_graph(damped_oscillations2, color=BLUE, x_min=0, x_max=7)
        else:
            graph = self.get_graph(damped_oscillations3, color=BLUE, x_min=0, x_max=7)

        anims.append(ShowCreation(graph, run_time=14, rate_func=linear))

        if hasattr(image, "fade_in_anim"):
            anims.append(image.fade_in_anim)
        else:
            anims.append(FadeIn(image))

        if hasattr(image, "continual_animations"):
            self.add(*image.continual_animations)

        self.play(*anims)
        self.play(FadeOut(graph), image.fade_out_anim)

    def get_physics_image(self, case):
        t_max = 6.5  # number of windings
        r = 0.4  # overlap of the windings //simulated view angle

        spring = ParametricCurve(
            lambda t: op.add(
                r * (np.sin(TAU * t) * RIGHT + (np.cos(TAU * t) - 1) * UP),
                t * DOWN,
            ),
            t_min=0, t_max=t_max,
            color=WHITE,
            stroke_width=2
        )
        spring.move_to(LEFT * 3 + DOWN * 6.5)
        spring.set_stroke(YELLOW, 2)
        spring.stretch(2, 1)

        weight = Square()
        weight.set_stroke(width=2)
        weight.set_color(RED)
        weight.set_fill(opacity=1, color=GREY)
        weight.set_height(0.5)
        weight.stretch(2, 0)

        point = Circle(radius=0.1)
        point.set_color(RED)
        point.set_fill(opacity=1, color=GREY)
        point.move_to(self.origin)

        line = Line(weight.get_center(), point.get_center())

        t_tracker = ValueTracker(0)

        group = VGroup(spring, weight, point)

        if case == 1:
            group.continual_animations = [
                t_tracker.add_updater(
                    lambda tracker, dt: tracker.set_value(
                        tracker.get_value() + dt / 2
                    )
                ),
                spring.add_updater(
                    lambda s: s.stretch_to_fit_height(
                        2 + 1.5 * np.cos(TAU * t_tracker.get_value()) * np.exp(-0.1 * t_tracker.get_value()),
                        about_edge=UP
                    )
                ),
                weight.add_updater(
                    lambda w: w.move_to(spring.get_bottom() + DOWN * weight.get_height() / 2)
                ),
                point.add_updater(
                    lambda p: p.move_to(
                        self.coords_to_point(t_tracker.get_value(), damped_oscillations1(t_tracker.get_value())))
                )
            ]
        elif case == -1:
            group.continual_animations = [
                t_tracker.add_updater(
                    lambda tracker, dt: tracker.set_value(
                        tracker.get_value() + dt / 2
                    )
                ),
                spring.add_updater(
                    lambda s: s.stretch_to_fit_height(
                        2 + 1.5 * np.exp(-0.1 * t_tracker.get_value()),
                        about_edge=UP
                    )
                ),
                weight.add_updater(
                    lambda w: w.move_to(spring.get_bottom() + DOWN * weight.get_height() / 2)
                ),
                point.add_updater(
                    lambda p: p.move_to(
                        self.coords_to_point(t_tracker.get_value(), damped_oscillations2(t_tracker.get_value())))
                )
            ]
        else:
            group.continual_animations = [
                t_tracker.add_updater(
                    lambda tracker, dt: tracker.set_value(
                        tracker.get_value() + dt / 2
                    )
                ),
                spring.add_updater(
                    lambda s: s.stretch_to_fit_height(
                        2 + 1.5 * (1 + TAU * t_tracker.get_value()) * np.exp(- TAU * t_tracker.get_value()),
                        about_edge=UP
                    )
                ),
                weight.add_updater(
                    lambda w: w.move_to(spring.get_bottom() + DOWN * weight.get_height() / 2)
                ),
                point.add_updater(
                    lambda p: p.move_to(
                        self.coords_to_point(t_tracker.get_value(), damped_oscillations3(t_tracker.get_value())))
                )
            ]

        def update_group_style(alpha):
            spring.set_stroke(width=2 * alpha)
            weight.set_fill(opacity=alpha)
            weight.set_stroke(width=2 * alpha)
            point.set_fill(opacity=alpha)
            point.set_stroke(width=2 * alpha)

        group.fade_in_anim = UpdateFromAlphaFunc(
            group,
            lambda g, a: update_group_style(a)
        )
        group.fade_out_anim = UpdateFromAlphaFunc(
            group,
            lambda g, a: update_group_style(1 - a)
        )

        return group


class TheFull(Scene):
    def construct(self):
        title = Title("Modelling the full soccerball")
        title.to_edge(UP)

        self.play(FadeIn(title))
        self.wait(90)


class Triangle(Scene):
    def __init__(self, **kwargs):
        Scene.__init__(self, x_min=-7,
                       x_max=7,
                       y_min=-4,
                       y_max=4,
                       y_axis_height=8,
                       x_axis_width=14,
                       y_axis_label="",
                       x_axis_label="",
                       graph_origin=DOWN * 4 + LEFT * 7,
                       axes_color="BLACK",
                       always_update_mobjects=True,
                       always_continually_update=True,
                       **kwargs)

    def construct(self):
        title = Title("Modelling the faces of the soccerball")
        title.to_edge(UP)

        self.play(FadeIn(title))
        self.setup_axes()

        radius = 2
        points = []
        xs = []
        ys = []
        for i in range(0, 3):
            xs.append(radius * np.cos(TAU / 3 * i) + 7)
            ys.append(radius * np.sin(TAU / 3 * i) + 3.5)
            p = Circle(radius=0.1)
            p.set_fill(color=BLUE, opacity=1)
            p.move_to(self.coords_to_point(xs[i], ys[i]))
            points.append(p)

        lines = []
        pairs = []
        for i in range(0, 3):
            for j in range(i + 1, 3):
                winding = 10
                r = 0.2
                offset = 0.5
                if j == 2 and i == 0:
                    line = Spring(points[j].get_center(), points[i].get_center(), winding, r,
                                  offset)  # change orientation of the last outside spring
                    pairs.append((j, i))
                else:
                    line = Spring(points[i].get_center(), points[j].get_center(), winding, r, offset)
                    pairs.append((i, j))
                line.set_color(YELLOW)
                lines.append(line)

        t_tracker = ValueTracker(0)

        arrow = DoubleArrow(points[1].get_center(), points[2].get_center())
        arrow.set_color(GREEN)
        arrow.shift(LEFT)
        d = Tex("d")
        d.set_color(GREEN)
        d.next_to(arrow, LEFT)

        force1 = Arrow(points[1].get_center(), points[1].get_center() + UP).scale(2)
        force1.set_fill(opacity=1, color=RED)

        force2 = Arrow(points[2].get_center(), points[2].get_center() + DOWN).scale(2)
        force2.set_fill(opacity=1, color=RED)

        group = VGroup(*lines, *points)
        group1 = VGroup(*lines, *points, force1, force2)

        group1.continual_animations1 = [
            t_tracker.add_updater(
                lambda tracker, dt: tracker.set_value(t_tracker.get_value() + dt)
            ),
            points[1].add_updater(
                lambda l: l.move_to(
                    self.coords_to_point(xs[1], ys[1]) + 0.3 * UP * t_tracker.get_value() / 5)
            ),
            points[2].add_updater(
                lambda l: l.move_to(
                    self.coords_to_point(xs[2], ys[2]) - 0.3 * UP * t_tracker.get_value() / 5)
            ),
            points[0].add_updater(
                lambda l: l.move_to(self.coords_to_point(xs[0], ys[0]))  # just redraw on top of the springs
            ),
            force1.add_updater(
                lambda f: f.move_to(points[1].get_center() + UP * f.get_height() / 2)
            ),
            force2.add_updater(
                lambda f: f.move_to(points[2].get_center() + DOWN * f.get_height() / 2)
            ),
        ]

        def update_line(p_line):
            index = lines.index(p_line)
            new_line = Spring(points[pairs[index][0]].get_center(), points[pairs[index][1]].get_center(), 10, 0.2, 0.5)
            new_line.set_color(p_line.get_color())
            lines[index].become(new_line)

        # self.add(group)
        self.play(FadeIn(group))
        self.wait(3)
        self.play(GrowFromCenter(arrow), Write(d))
        self.add(*group1.continual_animations1)
        self.play(GrowArrow(force1), GrowArrow(force2),
                  *[UpdateFromFunc(lines[i], update_line) for i in range(0, len(lines))], run_time=5)
        self.play(Uncreate(force1), Uncreate(force2), Uncreate(arrow), Uncreate(d), run_time=0.1)
        self.remove(*group1.continual_animations1)
        self.play(FadeOut(group), run_time=0)
        self.play(FadeOut(group1), run_time=0)

        t_tracker = ValueTracker(0)
        group2 = VGroup(*lines, *points)
        group2.continual_animations2 = [
            t_tracker.add_updater(
                lambda t, dt: t.set_value(t_tracker.get_value() + dt)
            ),
            points[1].add_updater(
                lambda l: l.move_to(
                    self.coords_to_point(xs[1], ys[1]) + 0.3 * UP * np.cos(
                        TAU * t_tracker.get_value()) * np.exp(
                        -0.6 * t_tracker.get_value()))
            ),
            points[2].add_updater(
                lambda l: l.move_to(
                    self.coords_to_point(xs[2], ys[2]) - 0.3 * UP * np.cos(
                        TAU * t_tracker.get_value()) * np.exp(
                        -0.6 * t_tracker.get_value()))
            ),
            points[0].add_updater(
                lambda l: l.move_to(self.coords_to_point(xs[0], ys[0]))  # just redraw on top of the springs
            ),
        ]

        self.add(*group2.continual_animations2)
        self.play(*[UpdateFromFunc(lines[i], update_line) for i in range(0, len(lines))], run_time=5)
        self.wait(10)
        self.remove(FadeOut(group2), FadeOut(arrow), FadeOut(d), FadeOut(group))
        self.remove(*group2.continual_animations2)
        self.play(FadeOut(group2), run_time=1)

        radius = 1.5
        points = []
        xs = []
        ys = []
        for i in range(0, 6):
            xs.append(radius * np.cos(TAU / 6 * i) + 7)
            ys.append(radius * np.sin(TAU / 6 * i) + 4)
            p = Circle(radius=0.1)
            p.set_fill(color=BLUE, opacity=1)
            p.move_to(self.coords_to_point(xs[i], ys[i]))
            points.append(p)

        lines = []
        pairs = []
        for i in range(0, 6):
            for j in range(i + 1, 6):
                if j == i + 1 or j - i == 5:
                    winding = 5
                    r = 0.1
                    color = YELLOW
                    offset = 0.1
                else:
                    winding = 10
                    r = 0.05
                    color = GREY
                    offset = 0.3
                if j == 5 and i == 0:
                    line = Spring(points[j].get_center(), points[i].get_center(), winding, r,
                                  offset)  # change orientation of the last outside spring
                else:
                    line = Spring(points[i].get_center(), points[j].get_center(), winding, r, offset)
                line.set_color(color)
                lines.append(line)
                pairs.append((i, j))

        hexagon = VGroup(*lines, *points)
        self.play(FadeIn(hexagon))
        self.play(ApplyMethod(hexagon.move_to, UP * 1 + RIGHT * 5.5))

        points = []
        xs = []
        ys = []
        for i in range(0, 5):
            xs.append(radius * np.cos(TAU / 5 * i) + 7)
            ys.append(radius * np.sin(TAU / 5 * i) + 4)
            p = Circle(radius=0.1)
            p.set_fill(color=BLUE, opacity=1)
            p.move_to(self.coords_to_point(xs[i], ys[i]))
            points.append(p)

        lines = []
        pairs = []
        for i in range(0, 5):
            for j in range(i + 1, 5):
                if j == i + 1 or j - i == 4:
                    winding = 5
                    r = 0.1
                    color = YELLOW
                    offset = 0.3
                else:
                    winding = 10
                    r = 0.05
                    color = GREY
                    offset = 0.6
                if j == 4 and i == 0:
                    line = Spring(points[j].get_center(), points[i].get_center(), winding, r,
                                  offset)  # change orientation of the last outside spring
                else:
                    line = Spring(points[i].get_center(), points[j].get_center(), winding, r, offset)
                line.set_color(color)
                lines.append(line)
                pairs.append((i, j))

        pentagon = VGroup(*lines, *points)
        self.play(FadeIn(pentagon))
        self.play(ApplyMethod(pentagon.move_to, DOWN * 2 + RIGHT * 2))
        self.wait(60)


class Code(Scene):
    def construct(self):
        title = Title("The Acceleration")
        title.to_edge(UP)
        title.move_to(LEFT * 4 + UP * 3.5)
        self.play(FadeIn(title))

        # grid = NumberPlane()
        # self.add(grid)

        # some tricks are used to get the scaling and positions of the subimages correctly
        # The reference is (-2,3.5) for the UL corner, the number of pixels is divided by 150

        image = ImageMobject("soccerball_code")
        image.rescale_to_fit(10.93 / 1.5, 1)
        image.move_to(RIGHT * 2 + UP * 3.5, UL)
        self.play(FadeIn(image))

        i_gravity = ImageMobject("gravity")
        i_gravity.rescale_to_fit(4.8 / 1.5, 0)
        i_gravity.move_to(RIGHT * (2 + 30 / 150) + UP * (3.5 - 251 / 150),
                          UL)  # means that the subimage is cut out at (30/251)

        i_floor = ImageMobject("floor")
        i_floor.rescale_to_fit(5.77 / 1.5, 0)
        i_floor.move_to(RIGHT * (2 + 32 / 150) + UP * (3.5 - 317 / 150),
                        UL)  # means that the subimage is cut out at (32/150)

        i_neighbour = ImageMobject("neighbour")
        i_neighbour.rescale_to_fit(6.90 / 1.5, 0)
        i_neighbour.move_to(RIGHT * (2 + 33 / 150) + UP * (3.5 - 465 / 150),
                            UL)  # means that the subimage is cut out at (41/254)

        i_air = ImageMobject("air")
        i_air.rescale_to_fit(7.14 / 1.5, 0)
        i_air.move_to(RIGHT * (2 + 32 / 150) + UP * (3.5 - 803 / 150),
                      UL)

        i_sum = ImageMobject("sum")
        i_sum.rescale_to_fit(5.19 / 1.5, 0)
        i_sum.move_to(RIGHT * (2 + 30 / 150) + UP * (3.5 - 1034 / 150),
                      UL)

        self.play(FadeIn(i_gravity), FadeIn(i_floor), FadeIn(i_neighbour), FadeIn(i_air), FadeIn(i_sum))
        self.wait(3)

        f_gravity = Tex(r"\vec F = m\cdot \vec g \Longrightarrow \vec a_\text{gr} = \vec g")
        a_gravity = Tex(r"\vec a_\text{gr} = \vec g")
        f_gravity.set_color(YELLOW)
        f_gravity.move_to(6.5 * LEFT + UP * 3, UL)
        a_gravity.set_color(YELLOW)
        a_gravity.move_to(6.5 * LEFT + UP * 3, UL)

        self.play(ApplyMethod(i_gravity.move_to, LEFT * 6 + UP * 2, UL))
        self.play(i_gravity.scale, 2,
                  i_gravity.move_to, LEFT * 6.5 + UP * 2, UL,
                  Write(f_gravity)
                  )

        self.wait(5)

        self.play(FadeOut(i_gravity), FadeOut(f_gravity),
                  Write(a_gravity))
        self.wait(5)

        f_floor = Tex(
            r"\vec F = -k_\text{fl}\cdot \vec h  \Longrightarrow \vec a_\text{fl} = \frac{k_\text{fl}}{m} h")
        a_floor = Tex(r"\vec a_\text{fl} = -\frac{k_\text{fl}}{m} \vec h_\perp")
        f_floor.set_color(GREEN)
        f_floor.move_to(6.5 * LEFT + UP * 2.2, UL)
        a_floor.set_color(GREEN)
        a_floor.move_to(6.5 * LEFT + UP * 2.5, UL)

        self.play(ApplyMethod(i_floor.move_to, LEFT * 6.5 + UP * 1, UL))
        self.play(i_floor.scale, 2,
                  i_floor.move_to, LEFT * 6.5 + UP * 1, UL,
                  Write(f_floor)
                  )

        self.wait(5)

        self.play(FadeOut(i_floor), FadeOut(f_floor),
                  Write(a_floor))
        self.wait(5)

        f_neighbour = Tex(
            r"\vec F_n = -k_\text{n}\cdot (\vec r-\vec r_\text{n}-\vec l_\text{n})-b \cdot (\vec v-\vec v_\text{n})")
        a_neighbour = Tex(
            r"\vec a_\text{n} = -\frac{k_\text{n}}{m} (\Delta\vec r_\text{n}-\vec l_\text{n})-\frac{b}{m}\cdot \Delta\vec v_\text{n}")
        f_neighbour.set_color(RED)
        f_neighbour.move_to(6.5 * LEFT + UP * 1, UL)
        a_neighbour.set_color(RED)
        a_neighbour.move_to(6.5 * LEFT + UP * 1.3, UL)

        self.play(ApplyMethod(i_neighbour.move_to, LEFT * 6.5, UL))
        self.play(i_neighbour.scale, 1.7,
                  i_neighbour.move_to, LEFT * 6.5, UL,
                  Write(f_neighbour)
                  )

        self.wait(5)

        self.play(FadeOut(i_neighbour), FadeOut(f_neighbour),
                  Write(a_neighbour))
        self.wait(5)

        a_air = Tex(r"\vec a_\text{nn} = -\frac{k_\text{air}}{m} (\Delta\vec r_\text{nn}-\vec l_\text{nn})")
        a_air.set_color(BLUE)
        a_air.move_to(6.5 * LEFT + UP * 0.1, UL)

        self.play(ApplyMethod(i_air.move_to, LEFT * 6.5 + DOWN * 1, UL))
        self.play(i_air.scale, 1.7,
                  i_air.move_to, LEFT * 6.5 + DOWN * 1, UL,
                  Write(a_air)
                  )

        self.wait(15)

        self.play(FadeOut(i_air))
        self.wait(5)

        a_sum = Tex(
            r"\vec a_\text{full} = \vec a_\text{gr}+\vec a_\text{fl}+\sum_\text{n} \vec a_\text{n}+\sum_\text{nn} \vec a_\text{nn}+\vec a_\text{fr}")
        a_sum.set_color(WHITE)
        a_sum.scale(0.95)
        a_sum.move_to(6.5 * LEFT + DOWN * 1.1, UL)

        self.play(ApplyMethod(i_sum.move_to, LEFT * 6.5 + DOWN * 2.5, UL))
        self.play(i_sum.scale, 2.,
                  i_sum.move_to, LEFT * 6.5 + DOWN * 2.5, UL,
                  Write(a_sum)
                  )

        self.wait(5)

        self.play(FadeOut(i_sum))
        self.play(a_sum.scale, 1.05,
                  a_sum.shift, DOWN * 0.5 + LEFT * 0.1)
        frame = Rectangle()
        frame.set_color(RED)
        frame.surround(a_sum)
        frame.stretch(0.3, 1)
        self.play(GrowFromCenter(frame))
        self.wait(15)


class Code2(Scene):
    def construct(self):
        title = Title("The Physics Engine")
        title.to_edge(UP)
        title.move_to(LEFT * 4 + UP * 3.5)
        self.play(FadeIn(title))

        i_engine = ImageMobject("engine")
        i_engine.scale(1.5)
        i_engine.shift(DOWN * 2)

        vel = Tex(r"\vec v_\text{new}=\vec v_\text{old}+\vec a_\text{full}\cdot dt")
        pos = Tex(r"\vec r_\text{new}=\vec r_\text{old}+\vec v_\text{new}\cdot dt")
        vel.set_color(YELLOW)
        pos.set_color(YELLOW)

        pos.next_to(i_engine, UP, LEFT).shift(RIGHT * pos.get_width() / 2 + UP * 1 + LEFT * i_engine.get_width() / 2)
        vel.next_to(pos, UP, LEFT).shift(RIGHT * vel.get_width() / 2 + UP * 0.5 + LEFT * pos.get_width() / 2)

        self.play(FadeIn(i_engine), Write(vel))
        self.play(Write(pos))
        self.wait(30)


class Spring(TipableVMobject):
    CONFIG = {
        "buff": 0,
        "path_arc": None,  # angle of arc specified here
    }

    def __init__(self, start=LEFT, end=RIGHT, windings=5, r=0.2, offset=0.05, **kwargs):
        digest_config(self, kwargs)
        self.set_start_and_end_attrs(start, end)
        self.windings = windings
        self.r = r
        self.offset = offset
        VMobject.__init__(self, **kwargs)

    def generate_points(self):
        """
        author: p6majo

        this is my crude first python implementation of a spring into the manim framework
        """
        dist = self.end - self.start
        length = np.sqrt(np.dot(dist, dist))
        t_max = self.windings + 0.5

        start = Line(LEFT * 0, DOWN * self.offset)
        curve = ParametricCurve(
            lambda t: op.add(
                self.r * (np.sin(TAU * t) * RIGHT + (np.cos(TAU * t) - 1) * UP),
                DOWN * (self.offset + 3 * t / t_max),
            ),
            t_min=0, t_max=t_max
        )
        end = Line(curve.get_bottom(), curve.get_bottom() + DOWN * self.offset)

        spring = VGroup(start, curve, end)
        spring.set_color(self.color)
        spring.stretch(length / spring.get_height(), 1)
        dot = np.dot(dist, UP) / length
        angle = np.arccos(dot)
        spring.move_to(self.end - dist * 0.5)
        spring.rotate(angle)
        spring_points=  spring.get_all_points();
        test1 = spring_points[0]-spring_points[len(spring_points)-1]
        test2 = test1-dist
        test3 = np.dot(test2,test2)
        if test3>0.1:
            spring.rotate(-2*angle)
        self.set_points(spring.get_all_points())

    def account_for_buff(self):
        if self.buff == 0:
            return
        #
        if self.path_arc == 0:
            length = self.get_length()
        else:
            length = self.get_arc_length()
        #
        if length < 2 * self.buff:
            return
        buff_proportion = self.buff / length
        self.pointwise_become_partial(
            self, buff_proportion, 1 - buff_proportion
        )
        return self

    def set_start_and_end_attrs(self, start, end):
        # If either start or end are Mobjects, this
        # gives their centers
        rough_start = self.pointify(start)
        rough_end = self.pointify(end)
        vect = normalize(rough_end - rough_start)
        # Now that we know the direction between them,
        # we can the appropriate boundary point from
        # start and end, if they're mobjects
        self.start = self.pointify(start, vect)
        self.end = self.pointify(end, -vect)

    def pointify(self, mob_or_point, direction=None):
        if isinstance(mob_or_point, Mobject):
            mob = mob_or_point
            if direction is None:
                return mob.get_center()
            else:
                return mob.get_boundary_point(direction)
        return np.array(mob_or_point)

    def put_start_and_end_on(self, start, end):
        curr_start, curr_end = self.get_start_and_end()
        if np.all(curr_start == curr_end):
            # TODO, any problems with resetting
            # these attrs?
            self.start = start
            self.end = end
            self.generate_points()
        return super().put_start_and_end_on(start, end)

    def get_vector(self):
        return self.get_end() - self.get_start()

    def get_unit_vector(self):
        return normalize(self.get_vector())

    def get_angle(self):
        return angle_of_vector(self.get_vector())

    def get_slope(self):
        return np.tan(self.get_angle())

    def set_angle(self, angle):
        self.rotate(
            angle - self.get_angle(),
            about_point=self.get_start(),
        )

    def set_length(self, length):
        self.scale(length / self.get_length())

    def set_opacity(self, opacity, family=True):
        # Overwrite default, which would set
        # the fill opacity
        self.set_stroke(opacity=opacity)
        if family:
            for sm in self.submobjects:
                sm.set_opacity(opacity, family)
        return self