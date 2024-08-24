#!/usr/bin/env python

from manimlib import *
import numpy as np

class Intro(Scene):
    def construct(self):
        title = TexText("Larger than Infinity", tex_to_color_map={"Larger": YELLOW, "Infinity": RED})
        title.scale(2)
        title.move_to(UP * 3.5)

        liste = CustomizedBulletedList("What are countable sets?", "Why are real numbers uncountable?",
                                       "Larger than infinity by examples?")

        liste.set_style()
        self.play(
            FadeIn(title)
        )
        self.wait(6)

        self.play(
            Write(liste[0])
        )
        self.wait(8)
        self.play(
            Write(liste[1])
        )
        self.wait(17)
        self.play(
            Write(liste[2])
        )
        self.wait(8)


class pos(object):
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y


class FiniteCountableSets(Scene):
    def construct(self):
        title = TexText("What are finite sets?", tex_to_color_map={"finite": RED})
        title.move_to(UP * 3.75)
        positions = []
        number_count = 14
        for i in range(0, number_count):
            p = pos(i - 6.5, 3)
            positions.append(p)

        numbers = []
        for i in range(0, number_count):
            if i < 9:
                text = Tex("0", str(i + 1), tex_to_color_map={"0": BLACK})
                numbers.append(text)
            else:
                numbers.append(TexText(str(i + 1)))

        self.play(
            FadeIn(title)
        )

        for i in range(0, 14):
            self.play(
                Write(numbers[i], run_time=0.1)
            )
            self.play(
                ApplyMethod(numbers[i].move_to, RIGHT * positions[i].x + UP * positions[i].y, rate_func=smooth,
                            run_time=0.1)
            )

        self.wait(3)

        boxes = []
        for i in range(0, 14):
            box = SurroundingRectangle(numbers[i])
            self.add(box)
            boxes.append(box)

        self.play(
            *[ApplyMethod(_.move_to, _.get_bottom() + DOWN * 0.5, run_time=0.5) for _ in boxes],
        )

        finite_set = [
            Tex(r"\{"),
            Tex("A"),
            Tex(","),
            Tex("B"),
            Tex(","),
            Tex("C"),
            Tex(r"\}")
        ]
        for i in range(1,len(finite_set)):
            finite_set[i].next_to(finite_set[i-1],RIGHT)

        for i in {2,4}:
            finite_set[i].shift(0.25*DOWN)

        finite_set2 = [
            Tex(r"\{"),
            Tex("a"),
            Tex(","),
            Tex("b"),
            Tex(","),
            Tex("c"),
            Tex(","),
            Tex("d"),
            Tex(","),
            Tex("e"),
            Tex(r"\}")
        ]

        finite_set2[0].next_to(finite_set[0],DOWN)

        for i in range(1, len(finite_set2)):
            finite_set2[i].next_to(finite_set2[i - 1], RIGHT)
        for i in {2,4,6,8}:
            finite_set2[i].shift(0.25*DOWN)

        for part in finite_set:
            part.set_color(YELLOW)
        for part in finite_set2:
            part.set_color(GREEN)

        self.play(
            *[Write(finite_set[i]) for i in range(0,len(finite_set))]
        )
        self.wait(1)

        for i in range(0, 3):
            mover = finite_set[2*i+1].copy()
            mover.generate_target()
            mover.target.move_to(boxes[i].get_center())
            self.play(
                MoveToTarget(mover, run_time=1)
            )

        boxes2 = []
        for i in range(0, len(boxes)):
            box = boxes[i].copy()
            box.set_color(GREEN)
            boxes2.append(box)

        self.wait(3)

        self.play(
            *[Write(finite_set2[i]) for i in range(0, len(finite_set2))]
        )

        self.play(
            *[ApplyMethod(boxes2[_].move_to, boxes[_].get_bottom() + DOWN * 0.5, run_time=0.5) for _ in range(0, 14)],
        )

        for i in range(0, 5):
            mover = finite_set2[2 * i + 1].copy()
            mover.generate_target()
            mover.target.move_to(boxes2[i].get_center())
            self.play(
                MoveToTarget(mover, run_time=1)
            )

        self.wait(7)


class InfiniteCountableSets(Scene):
    def construct(self):
        title = TexText("What are countable sets?", tex_to_color_map={"countable": RED})
        title.move_to(UP * 3.75)
        positions = []
        l = 100
        for i in range(0, l):
            p = pos(i - 6.5, 3)
            positions.append(p)

        numbers = []
        for i in range(0, l):
            if i < 9:
                text = Tex("0", str(i + 1), tex_to_color_map={"0": BLACK})
                numbers.append(text)
            else:
                numbers.append(TexText(str(i + 1)))

        self.play(
            FadeIn(title)
        )

        for i in range(0, l):
            numbers[i].move_to(RIGHT * positions[i].x + UP * positions[i].y)
            self.play(
                Write(numbers[i], run_time=0.0)
            )

        self.wait(10)

        boxes = []
        for i in range(0, l):
            box = SurroundingRectangle(numbers[i])
            self.add(box)
            boxes.append(box)

        self.play(
            *[ApplyMethod(_.move_to, _.get_bottom() + DOWN * 0.5, run_time=0.5) for _ in boxes]
        )

        self.wait(20)

        finite_set = Tex(r"\{n\in \mathbb{N}| n \text{ even}\}")
        finite_set.set_color(YELLOW)

        finite_set2 = Tex(r"\{n\in \mathbb{N}| n \text{ multiples of 4}\}")
        finite_set2.set_color(GREEN)
        finite_set2.move_to(DOWN * 0.75)

        self.play(
            Write(finite_set)
        )
        self.wait(1)

        evens = []
        for i in range(0, l):
            even = TexText(str(2 * (i + 1)))
            if 2 * (i + 1) > 99:
                even.scale(0.7)
            even.set_color(YELLOW)
            even.move_to(boxes[i].get_center())
            evens.append(even)
            self.play(
                Write(even, run_time=0.01)
            )

        boxes2 = []
        for i in range(0, len(boxes)):
            box = boxes[i].copy()
            box.set_color(GREEN)
            boxes2.append(box)

        self.wait(3)

        self.play(
            *[ApplyMethod(boxes[_].move_to, boxes[_].get_center() + LEFT * 30, run_time=10) for _ in range(0, l)],
            *[ApplyMethod(evens[_].move_to, evens[_].get_center() + LEFT * 30, run_time=10) for _ in range(0, l)],
            *[ApplyMethod(numbers[_].move_to, numbers[_].get_center() + LEFT * 30, run_time=10) for _ in range(0, l)]
        )

        self.wait(1)

        self.play(
            *[ApplyMethod(boxes[_].move_to, boxes[_].get_center() + RIGHT * 30, run_time=5) for _ in range(0, l)],
            *[ApplyMethod(evens[_].move_to, evens[_].get_center() + RIGHT * 30, run_time=5) for _ in range(0, l)],
            *[ApplyMethod(numbers[_].move_to, numbers[_].get_center() + RIGHT * 30, run_time=5) for _ in range(0, l)]
        )

        self.wait(3)

        self.play(
            Write(finite_set2)
        )
        self.wait(1)

        self.play(
            *[ApplyMethod(boxes2[_].move_to, boxes[_].get_bottom() + DOWN * 0.5, run_time=0.5) for _ in range(0, l)],
        )

        fours = []
        for i in range(0, l):
            even = TexText(str(4 * (i + 1)))
            if 4 * (i + 1) > 99:
                even.scale(0.7)
            even.set_color(GREEN)
            even.move_to(boxes2[i].get_center())
            evens.append(even)
            self.play(
                Write(even, run_time=0.01)
            )
            fours.append(even)

        self.wait(1)

        self.play(
            *[ApplyMethod(boxes[_].move_to, boxes[_].get_center() + LEFT * 30, run_time=10) for _ in range(0, l)],
            *[ApplyMethod(boxes2[_].move_to, boxes2[_].get_center() + LEFT * 30, run_time=10) for _ in range(0, l)],
            *[ApplyMethod(evens[_].move_to, evens[_].get_center() + LEFT * 30, run_time=10) for _ in range(0, l)],
            *[ApplyMethod(numbers[_].move_to, numbers[_].get_center() + LEFT * 30, run_time=10) for _ in range(0, l)],
            *[ApplyMethod(fours[_].move_to, fours[_].get_center() + LEFT * 30, run_time=10) for _ in range(0, l)]
        )

        self.wait(1)

        self.play(
            *[ApplyMethod(boxes[_].move_to, boxes[_].get_center() + RIGHT * 30, run_time=5) for _ in range(0, l)],
            *[ApplyMethod(boxes2[_].move_to, boxes2[_].get_center() + RIGHT * 30, run_time=5) for _ in range(0, l)],
            *[ApplyMethod(evens[_].move_to, evens[_].get_center() + RIGHT * 30, run_time=5) for _ in range(0, l)],
            *[ApplyMethod(numbers[_].move_to, numbers[_].get_center() + RIGHT * 30, run_time=5) for _ in range(0, l)],
            *[ApplyMethod(fours[_].move_to, fours[_].get_center() + RIGHT * 30, run_time=5) for _ in range(0, l)]
        )

        self.wait(20)


class Fractions(Scene):
    def construct(self):
        title = TexText("What are countable sets?", tex_to_color_map={"countable": RED})
        title.move_to(UP * 3.75)
        positions = []
        l = 150
        for i in range(0, l):
            p = pos(i - 6.5, 3)
            positions.append(p)

        numbers = []
        for i in range(0, l):
            if i < 9:
                text = Tex("0", str(i + 1), tex_to_color_map={"0": BLACK})
                numbers.append(text)
            else:
                numbers.append(TexText(str(i + 1)))

        self.play(
            FadeIn(title)
        )

        for i in range(0, l):
            numbers[i].move_to(RIGHT * positions[i].x + UP * positions[i].y)
            self.play(
                Write(numbers[i], run_time=0.0)
            )

        # for i in range(14, l):
        #     numbers[i].move_to(RIGHT * positions[i].x + UP * positions[i].y)
        #     self.play(
        #         Write(numbers[i], run_time=0.0)
        #     )

        self.wait(1)

        boxes = []
        for i in range(0, l):
            box = SurroundingRectangle(numbers[i])
            self.add(box)
            boxes.append(box)

        self.play(
            *[ApplyMethod(_.move_to, _.get_bottom() + DOWN * 0.75, run_time=0.5) for _ in boxes]
        )

        fractions = []
        for i in range(1, 20):
            sublist = []
            for j in range(0, i + 1):
                fraction = Tex(r"\frac{" + str(j) + "}{" + str(i) + "}")
                fraction.move_to(LEFT * 7.5 + RIGHT * i * 0.75 + UP * 1 + DOWN * j * 0.75)
                fraction.scale(0.5)
                sublist.append(fraction)
            fractions.append(sublist)

        boost = 1
        for fractionList in fractions:
            for fraction in fractionList:
                boost = boost + 0.1
                self.play(
                    Write(fraction, run_time=0.5 / boost)
                )

        self.wait(5)

        survivors = [[1, 1]]
        for i in range(2, 20):
            sublist = []
            for j in range(0, i + 1):
                if np.gcd(j, i) > 1:
                    sublist.append(0)
                else:
                    sublist.append(1)
            survivors.append(sublist)

        boost = 1
        crosses = [[]]  # no crosses in the first column
        for i in range(2, 20):
            sublist = []
            for j in range(0, i + 1):
                boost = boost + 0.1
                if survivors[i - 1][j] == 0:
                    cr = Cross(fractions[i - 1][j], color=RED)
                    sublist.append(cr)
                    self.play(FadeIn(cr), run_time=0.5 / boost)
            crosses.append(sublist)

        boost = 0.5
        for i in range(2, 20):
            cross_list = crosses[i - 1]
            for j in range(0, i + 1):
                boost = boost + 0.1
                if survivors[i - 1][j] == 0:
                    self.play(
                        FadeOut(cross_list.pop(0), run_time=0.001),
                        FadeOut(fractions[i - 1][j], run_time=0.001),
                        *[ApplyMethod(_.move_to, _.get_center() + UP * 0.75, run_time=0.5 / boost) for _ in cross_list],
                        *[ApplyMethod(fractions[i - 1][_].move_to, fractions[i - 1][_].get_center() + UP * 0.75,
                                      run_time=0.5 / boost) for _ in range(j + 1, i + 1)]
                    )

        self.play(
            *[ApplyMethod(_.stretch, 2, 1, run_time=0.5) for _ in boxes]
        )

        movers = []
        counter = 0
        boost = 1
        for i in range(1, 20):
            for j in range(0, i + 1):
                if survivors[i - 1][j] == 1:
                    if counter < 14:
                        boost = boost + 0.1
                    else:
                        boost = boost + 1
                    self.play(
                        ApplyMethod(fractions[i - 1][j].move_to, boxes[counter].get_center(), run_time=0.25 / boost)
                    )
                    self.play(
                        ApplyMethod(fractions[i - 1][j].scale, 1.4, run_time=0.25 / boost)
                    )
                    movers.append(fractions[i - 1][j])
                    counter = counter + 1

        self.play(
            *[ApplyMethod(boxes[_].move_to, boxes[_].get_center() + LEFT * 100, run_time=20) for _ in range(0, l)],
            *[ApplyMethod(numbers[_].move_to, numbers[_].get_center() + LEFT * 100, run_time=20) for _ in range(0, l)],
            *[ApplyMethod(movers[_].move_to, movers[_].get_center() + LEFT * 100, run_time=20) for _ in
              range(0, len(movers))]
        )

        self.play(
            *[ApplyMethod(boxes[_].move_to, boxes[_].get_center() + RIGHT * 100, run_time=5) for _ in range(0, l)],
            *[ApplyMethod(numbers[_].move_to, numbers[_].get_center() + RIGHT * 100, run_time=5) for _ in range(0, l)],
            *[ApplyMethod(movers[_].move_to, movers[_].get_center() + RIGHT * 100, run_time=5) for _ in
              range(0, len(movers))]
        )

        self.wait(5)


class Reals(Scene):
    def construct(self):
        title = TexText("Is every set countable?", tex_to_color_map={"countable": RED})
        title.move_to(UP * 3.75)
        positions = []
        l = 150
        for i in range(0, l):
            p = pos(i - 6.5, 3)
            positions.append(p)

        ints = []
        for i in range(0, l):
            if i < 9:
                text = Tex("0", str(i + 1), tex_to_color_map={"0": BLACK})
                ints.append(text)
            else:
                ints.append(TexText(str(i + 1)))

        self.play(
            FadeIn(title)
        )

        for i in range(0, l):
            ints[i].move_to(RIGHT * positions[i].x + UP * positions[i].y)
            self.play(
                Write(ints[i], run_time=0.0)
            )

        self.wait(1)

        boxes = []
        for i in range(0, l):
            box = SurroundingRectangle(ints[i])
            box.set_color(BLACK)
            self.add(box)
            boxes.append(box)

        self.play(
            *[ApplyMethod(_.move_to, _.get_bottom() + DOWN * 0.75, run_time=0.0) for _ in boxes],
        )
        self.play(
            *[ApplyMethod(_.stretch, 2, 1, run_time=0.5) for _ in boxes]
        )

        fractions = [Tex("0"), Tex("1")]
        fraction_values = [0, 1]
        for i in range(2, 20):
            sublist = []
            for j in range(1, i + 1):
                if np.gcd(i, j) == 1:
                    fraction = Tex(r"\frac{" + str(j) + "}{" + str(i) + "}")
                    fractions.append(fraction)
                    fraction_values.append(j / i)
                    fraction.scale(0.9)

        for i in range(0, len(fractions)):
            fractions[i].move_to(boxes[i].get_center())

        self.play(
            *[Write(_, run_time=0.1) for _ in fractions]
        )

        number_line = NumberLine(x_min=0, x_max=10)
        number_line.move_to(LEFT * 0+UP*0.1)

        blue = Color("blue")
        red = Color("red")

        colors = list(blue.range_to(red, 101))
        colors2 = list(red.range_to(blue, 101))

        color_line = Line(LEFT * 5, RIGHT * 5)
        color_line.set_color(colors2)

        self.play(
            FadeIn(number_line),
            #FadeIn(color_line)
        )

        arrows = []
        for i in range(0, 13):
            fraction = fractions[i].copy()
            color = colors[math.floor(fraction_values[i] * 100)]
            fraction.set_color(color)
            arrow = Arrow(Line(UP * 0.1, UP * 0))
            arrow.rotate(np.pi / 2)
            arrow.scale(0.5)
            position = LEFT * 5 + 10 * fraction_values[i] * RIGHT
            arrow.move_to(position + UP * 0.25)
            arrow.set_color(color)
            arrows.append(arrow)
            self.play(
                ApplyMethod(fractions[i].set_color, color, run_time=0.01),
                ApplyMethod(fraction.move_to, position + DOWN * 1, run_time=0.1),
                FadeIn(arrow, run_time=0.1)
            )

        self.play(
            *[ApplyMethod(boxes[_].move_to, boxes[_].get_center() + LEFT * 13, run_time=1) for _ in range(0, l)],
            *[ApplyMethod(ints[_].move_to, ints[_].get_center() + LEFT * 13, run_time=1) for _ in
              range(0, l)],
            *[ApplyMethod(fractions[_].move_to, fractions[_].get_center() + LEFT * 13, run_time=1) for _ in
              range(0, len(fractions))]
        )

        for i in range(13, len(fractions)):
            color = colors[math.floor(fraction_values[i] * 100)]
            arrow = Arrow(Line(UP * 0.1, UP * 0))
            arrow.rotate(-np.pi / 2)
            arrow.scale(0.2)
            position = LEFT * 5 + 10 * fraction_values[i] * RIGHT + UP * 0.1
            arrow.set_color(color)
            arrow.move_to(boxes[i].get_bottom())
            arrows.append(arrow)

        self.play(
            AnimationGroup(Succession(
                *[ApplyMethod(arrows[i].move_to, LEFT * 5 + 10 * fraction_values[i] * RIGHT + UP * 0) for i in
                  range(13, len(fractions))]), run_time=10),
            AnimationGroup(Succession(
                *[ApplyMethod(fractions[i].set_color, colors[math.floor(fraction_values[i] * 100)]) for i in
                  range(13, len(fractions))]), run_time=10),
            *[ApplyMethod(boxes[_].move_to, boxes[_].get_center() + LEFT * 86, run_time=12) for _ in range(0, l)],
            *[ApplyMethod(ints[_].move_to, ints[_].get_center() + LEFT * 86, run_time=12) for _ in range(0, l)],
            *[ApplyMethod(fractions[_].move_to, fractions[_].get_center() + LEFT * 86, run_time=12) for _ in
              range(0, len(fractions))]
        )

        self.play(
            *[ApplyMethod(boxes[_].move_to, boxes[_].get_center() + RIGHT * 100, run_time=1) for _ in range(0, l)],
            *[ApplyMethod(ints[_].move_to, ints[_].get_center() + RIGHT * 100, run_time=1) for _ in
              range(0, l)],
            *[ApplyMethod(fractions[_].move_to, fractions[_].get_center() + RIGHT * 100, run_time=1) for _ in
              range(0, len(fractions))]
        )


class Cantor(Scene):
    def construct(self):
        removables = []

        title = TexText("Is every set countable?", tex_to_color_map={"countable": RED})
        title.move_to(UP * 3.75)

        # store all objects that have to be removed eventually to clear the screen
        removables.append(title)

        positions = []
        number_count = 250  # size of the number list (first fractions, later real numbers)
        for i in range(0, number_count):
            p = pos(i - 6.5, 3)
            positions.append(p)

        numbers = []
        for i in range(0, number_count):
            if i < 9:
                # insert a black padding for single digit numbers
                # this should be down much smarter, I guess
                text = Tex("0", str(i + 1), tex_to_color_map={"0": BLACK})
                numbers.append(text)
            else:
                numbers.append(TexText(str(i + 1)))

        for i in range(0, number_count):
            numbers[i].move_to(RIGHT * positions[i].x + UP * positions[i].y)

        removables = removables + numbers

        boxes = []  # create the boxes
        for i in range(0, number_count):
            box = SurroundingRectangle(numbers[i])
            box.set_color(GREY)
            box.move_to(box.get_bottom() + DOWN * 0.75)
            box.stretch(2, 1)
            boxes.append(box)

        removables = removables + boxes

        # prepare the coloured number line, where the value of the number is represented by the color
        blue = Color("blue")
        red = Color("red")
        colors = list(blue.range_to(red, number_count + 1))
        colors2 = list(red.range_to(blue, number_count + 1))

        fractions = [Tex("0"), Tex("1")]  # create a list for the fraction objects
        values = [0, 1]  # create a list for the fraction values
        for i in range(2, 20):
            sublist = []
            for j in range(1, i + 1):
                if np.gcd(i, j) == 1:  # only consider fractions with co-prime numerator and denominator
                    fraction = Tex(r"\frac{" + str(j) + "}{" + str(i) + "}")
                    fractions.append(fraction)
                    values.append(j / i)
                    fraction.scale(0.9)

        # place the fraction objects into their boxes
        for i in range(0, len(fractions)):
            fractions[i].move_to(boxes[i].get_center())

        # prepare a list of boxes 
        arrows = []
        l2 = len(fractions)

        for i in range(0, l2):
            color = colors[math.floor(values[i] * number_count)]
            fractions[i].set_color(color)
            arrow = Arrow(Line(UP * 0.1, UP * 0))
            arrow.rotate(-np.pi / 2)
            if i < 13:
                arrow.scale(0.5)
                position = LEFT * 5 + 10 * values[i] * RIGHT
                arrow.move_to(position + UP * 0.25)
            else:
                arrow.scale(0.2)
                position = LEFT * 5 + 10 * values[i] * RIGHT + UP * 0.1
                arrow.move_to(position)
            arrow.set_color(color)
            arrows.append(arrow)

        number_line = NumberLine(x_min=0, x_max=10)
        number_line.move_to(LEFT * 0)

        color_line = Line(LEFT * 5, RIGHT * 5)
        color_line.set_color(colors2)

        fractions2 = []
        for i in range(0, 13):
            fractions2.append(fractions[i].copy())

        self.play(
            Write(title),
            *[Write(_, run_time=0.01) for _ in numbers],
            *[FadeIn(_, run_time=0.0) for _ in boxes],
            *[FadeIn(_, run_time=0.0) for _ in arrows],
            *[FadeIn(_, run_time=0.0) for _ in fractions],
            *[ApplyMethod(fractions2[i].move_to, LEFT * 5 + 10 * values[i] * RIGHT + DOWN * 1, run_time=0.0) for
              i in range(0, 13)],
            FadeIn(number_line),
            FadeIn(color_line)
        )

        self.wait(3)

        for i in range(0, len(fractions2)):
            fraction2 = fractions2.pop(0)
            fraction = fractions.pop(0)
            arrow = arrows.pop(0)
            self.play(
                FadeOut(fraction, run_time=0.1),
                FadeOut(fraction2, run_time=0.1),
                FadeOut(arrow, run_time=0.1)
            )

        self.play(
            FadeOut(color_line),
            *[FadeOut(_, run_time=0.1) for _ in arrows],
            *[FadeOut(_, run_time=0.1) for _ in fractions],
            *[ApplyMethod(_.stretch, 0.7, 1, run_time=0.5) for _ in boxes]
        )
        self.play(
            *[ApplyMethod(_.move_to, _.get_center() + UP * 0.25, run_time=0.1) for _ in boxes]
        )

        # select special seed to get a fixed random sequence of numbers that allows for nice
        # intervals and intervals of intervals
        random.seed(7)  # 7, 1
        line_sep = 0.75
        vals = []
        for i in range(0, number_count):
            val = random.random()
            vals.append(val)

        # create list of points that represents the real numbers
        points = []
        for i in range(0, number_count):
            color = colors[math.floor(vals[i] * number_count)]
            point = Circle(fill_color=color, fill_opacity=1)
            point.scale(0.03)
            point.move_to(LEFT * 5 + 10 * vals[i] * RIGHT)
            point.set_color(color)
            points.append(point)

        removables = removables + points

        self.play(
            *[Write(_, run_time=0.5) for _ in points]
        )

        self.wait()

        self.play(
            *[ApplyMethod(points[i].move_to, boxes[i].get_center(), run_time=2) for i in range(0, number_count)]
        )

        self.play(
            *[ApplyMethod(points[i].scale, 7, run_time=0.5) for i in range(0, number_count)]
        )

        self.wait(2)

        # calculate first interval
        if vals[0] < vals[1]:
            minimum = vals[0]
            maximum = vals[1]
        else:
            minimum = vals[1]
            maximum = vals[0]

        arrows = []
        for i in range(0, 2):
            if i == 0:
                val = minimum
            else:
                val = maximum
            arrow = Arrow(Line(UP * 0.1, UP * 0))
            arrow.rotate(-np.pi / 2)
            arrow.scale(0.5)
            position = LEFT * 5 + 10 * val * RIGHT + UP * 1
            arrow.move_to(position)
            arrow.set_color(colors[math.floor(val * number_count)])
            arrows.append(arrow)

        removables = removables + arrows
        removables.append(number_line)

        self.play(
            ApplyMethod(number_line.move_to, UP * 0.75, run_time=0.5),
            *[ApplyMethod(points[i].move_to, LEFT * 5 + 10 * vals[i] * RIGHT + UP * 0.75, run_time=2) for i in
              range(0, 2)]
        )

        self.play(
            *[ApplyMethod(points[i].scale, 0.2, run_time=0.5) for i in range(0, 2)]
        )

        selected = [0, 1]

        self.play(
            *[FadeIn(_, run_time=0.1) for _ in arrows]
        )
        self.wait(1)

        removables2 = self.next(2, 0, minimum, maximum, boxes, numbers, points, vals, selected)
        removables = removables + removables2

        self.wait(1)

        self.play(
            *[FadeOut(_) for _ in removables]
        )

    def shifting(self, boxes, numbers, points, shift, selected):
        """ shift the set of all boxes and numbers , skip selected fractions

        Parameters
        ----------
        boxes list of boxes
        numbers list of numbers (fractions or reals)
        points list of points
        shift the amount of shifting
        selected selected objects that are excluded from shifting

        Returns
        -------

        """
        movers = []

        for i in range(0, len(points)):
            if not (i in selected):
                movers.append(points[i])

        self.play(
            *[ApplyMethod(boxes[_].move_to, boxes[_].get_center() + LEFT * shift, run_time=shift / 10) for _ in
              range(0, len(boxes))],
            *[ApplyMethod(numbers[_].move_to, numbers[_].get_center() + LEFT * shift, run_time=shift / 10) for _ in
              range(0, len(numbers))],
            *[ApplyMethod(movers[_].move_to, movers[_].get_center() + LEFT * shift, run_time=shift / 10) for _ in
              range(0, len(movers))]
        )

    def next(self, current, level, mi, ma, boxes, numbers, points, vals, selected):
        """determine the next interval that lies inside the previous interval defined by mi and ma

        Parameters
        ----------
        current current position in the list of numbers
        level   current level
        mi left interval limit
        ma right interval limit
        boxes list of boxes
        numbers list of numbers
        points current position in the list
        vals values of all reals
        selected selected objects

        Returns
        -------

        """
        new_pos = []
        line_sep = 0.75
        blue = Color("blue")
        red = Color("red")
        current_old = current

        removables = []

        colors = list(blue.range_to(red, len(vals) + 1))

        # find two new points that lie inside the previous interval
        while len(new_pos) < 2 and current < len(vals):
            if mi < vals[current] < ma:
                new_pos.append(current)
            current = current + 1

        if len(new_pos) < 2:
            print("No new interval has been found. Use more numbers or change the seed.")

        if vals[new_pos[0]] < vals[new_pos[1]]:
            mi = vals[new_pos[0]]
            ma = vals[new_pos[1]]
        else:
            mi = vals[new_pos[1]]
            ma = vals[new_pos[0]]

        number_line = NumberLine(x_min=0, x_max=10)
        number_line.move_to(DOWN * level * line_sep)

        removables.append(number_line)

        shift = 0
        if new_pos[0] > 6:
            shift = new_pos[0] - current_old
            self.shifting(boxes, numbers, points, shift, selected)

        selected.append(new_pos[0])

        self.play(
            FadeIn(number_line, run_time=0.5),
            ApplyMethod(points[new_pos[0]].move_to, LEFT * 5 + 10 * vals[new_pos[0]] * RIGHT + DOWN * level * line_sep,
                        run_time=1)
        )

        self.play(
            ApplyMethod(points[new_pos[0]].scale, 0.2, run_time=0.25)
        )

        shift = (new_pos[1] - new_pos[0])
        self.shifting(boxes, numbers, points, shift, selected)

        selected.append(new_pos[1])
        self.play(
            ApplyMethod(points[new_pos[1]].move_to, LEFT * 5 + 10 * vals[new_pos[1]] * RIGHT + DOWN * level * line_sep,
                        run_time=0.5)
        )

        self.play(
            ApplyMethod(points[new_pos[1]].scale, 0.2, run_time=0.25)
        )

        arrows = []
        for i in range(0, 2):
            val = vals[new_pos[i]]
            arrow = Arrow(Line(UP * 0.1, UP * 0))
            arrow.rotate(-np.pi / 2)
            arrow.scale(0.5)
            position = LEFT * 5 + 10 * val * RIGHT + DOWN * level * line_sep + UP * 0.25
            arrow.move_to(position)
            arrow.set_color(colors[math.floor(val * len(vals))])
            arrows.append(arrow)

        removables = removables + arrows

        self.play(
            *[FadeIn(_, run_time=0.1) for _ in arrows]
        )

        if level < 2:
            self.wait()
            removables = removables + self.next(current, level + 1, mi, ma, boxes, numbers, points, vals, selected)
        else:
            print(mi)
            print(ma)

        return removables


class Proof(Scene):
    def construct(self):
        title = TexText("Cantor's proof of Uncountability", tex_to_color_map={"Uncountability": RED})
        title.move_to(UP * 3.75)

        final_points = []
        line_sep = 0.75
        l = 250

        final_line = NumberLine(x_min=0, x_max=10)
        final_line.move_to(LEFT * 0 + DOWN * (5 * line_sep + 0.25))

        blue = Color("blue")
        red = Color("red")

        colors = list(blue.range_to(red, l + 1))

        points = []
        positions = [0.5097908203084482, 0.5287201947298235]
        for i in range(0, 2):
            color = colors[math.floor(positions[i] * l)]
            point = Circle(fill_color=color, fill_opacity=1)
            point.scale(0.03)
            point.move_to(LEFT * 5 + 10 * positions[i] * RIGHT + DOWN * (5 * 0.75 + 0.25))
            point.set_color(color)
            points.append(point)

        arrows = []

        for i in range(0, 2):
            arrow = Arrow(Line(UP * 0.1, UP * 0))
            arrow.rotate(np.pi/2)
            arrow.scale(0.5)
            position = LEFT * 5 + 10 * positions[i] * RIGHT + DOWN * (5 * line_sep + 0.25) + UP * 0.25
            arrow.move_to(position)
            arrow.set_color(colors[math.floor(positions[i] * l)])
            arrows.append(arrow)

        self.play(
            FadeIn(final_line, run_time=0.0001),
            *[FadeIn(_, run_time=0.0001) for _ in points],
            *[FadeIn(_, run_time=0.0001) for _ in arrows],
            Write(title, run_time=1)
        )

        group = VGroup(final_line, *arrows, *points)
        self.play(
            ApplyMethod(group.move_to, UP * 2, run_time=2)
        )

        self.wait(2)
        final_line2 = NumberLine(x_min=5, x_max=6)
        final_line2.move_to(final_line.get_center() + RIGHT * 0.5)

        self.play(Transform(final_line, final_line2, run_time=1))
        self.play(
            ApplyMethod(final_line.stretch, 3, 0, run_time=1),
            ApplyMethod(arrows[0].move_to, arrows[0].get_center() + LEFT * 0.904, run_time=1),
            ApplyMethod(points[0].move_to, points[0].get_center() + LEFT * 0.904, run_time=1),
            ApplyMethod(arrows[1].move_to, arrows[1].get_center() + LEFT * 0.4256, run_time=1),
            ApplyMethod(points[1].move_to, points[1].get_center() + LEFT * 0.4256, run_time=1),
        )
        self.wait(3)

        newgroup = VGroup(final_line, *arrows, *points)
        newgroup1 = newgroup.copy()
        newgroup2 = newgroup.copy()
        newgroup3 = newgroup.copy()

        shift1 = newgroup.get_center() + LEFT * 5.5 + DOWN * 2
        shift2 = newgroup.get_center() + LEFT * 0.5 + DOWN * 2
        shift3 = newgroup.get_center() + RIGHT * 4.5 + DOWN * 2

        self.play(
            ApplyMethod(newgroup1.move_to, shift1, run_time=2),
            # ApplyMethod(newgroup2.move_to, shift2, run_time=2),
            # ApplyMethod(newgroup3.move_to, shift3, run_time=2)
        )

        self.wait(7)

        # first case

        arrows1 = []
        for i in range(0, 2):
            arrow = DashedVMobject(Arrow(DashedLine(UP * 0.1, UP * 0)))
            arrow.rotate(-np.pi / 2)
            #arrow.scale(0.75)
            position = arrows[i].get_center() + shift1 + DOWN *2.05 + LEFT * 0.5
            arrow.move_to(position)
            arrow.set_color(WHITE)
            arrows1.append(arrow)

        points1 = []
        for i in range(0, 2):
            point = Circle(fill_color=WHITE, fill_opacity=1)
            position = points[i].get_center() + shift1 + DOWN * 2 + LEFT * 0.5
            point.scale(0.03)
            point.move_to(position)
            point.set_color(WHITE)
            points1.append(point)

        self.play(
            ApplyMethod(arrows1[0].move_to, arrows1[0].get_center() + RIGHT * 0.2, run_time=0.5),
            ApplyMethod(arrows1[1].move_to, arrows1[1].get_center() + LEFT * 0.2, run_time=0.5),
            ApplyMethod(points1[0].move_to, points1[0].get_center() + RIGHT * 0.2, run_time=0.5),
            ApplyMethod(points1[1].move_to, points1[1].get_center() + LEFT * 0.2, run_time=0.5)
        )

        self.wait(3)

        points1b = []
        for i in range(0, 2):
            point = Circle(fill_color=RED, fill_opacity=1)
            position = (points1[0].get_center() + points1[1].get_center()) / 2 + LEFT * i * 0.05 + RIGHT * (
                        1 - i) * 0.05
            point.scale(0.03)
            point.move_to(position)
            point.set_color(RED)
            points1b.append(point)

        self.play(
            *[FadeIn(_, run_time=0.5) for _ in points1b]
        )

        # second case

        self.play(
            # ApplyMethod(newgroup1.move_to, shift1, run_time=2),
            ApplyMethod(newgroup2.move_to, shift2, run_time=2),
            # ApplyMethod(newgroup3.move_to, shift3, run_time=2)
        )

        arrows2 = []
        for i in range(0, 2):
            arrow = DashedVMobject(Arrow(DashedLine(UP * 0.1, UP * 0)))
            arrow.rotate(-np.pi / 2)
            #arrow.scale(0.5)
            position = arrows[i].get_center() + shift2 + DOWN * 2.05 + LEFT * 0.5
            arrow.move_to(position)
            arrow.set_color(WHITE)
            arrows2.append(arrow)

        points2 = []
        for i in range(0, 2):
            point = Circle(fill_color=WHITE, fill_opacity=1)
            position = points[i].get_center() + shift2 + DOWN * 2 + LEFT * 0.5
            point.scale(0.03)
            point.move_to(position)
            point.set_color(WHITE)
            points2.append(point)

        self.play(
            ApplyMethod(arrows2[0].move_to, arrows2[0].get_center() + RIGHT * 0.2, run_time=10),
            ApplyMethod(arrows2[1].move_to, arrows2[1].get_center() + LEFT * 0.2, run_time=10),
            ApplyMethod(points2[0].move_to, points2[0].get_center() + RIGHT * 0.2, run_time=10),
            ApplyMethod(points2[1].move_to, points2[1].get_center() + LEFT * 0.2, run_time=10)
        )

        self.wait(3)

        points2b = []
        for i in range(0, 2):
            point = Circle(fill_color=RED, fill_opacity=1)
            position = (points2[0].get_center() + points2[1].get_center()) / 2 + LEFT * i * 0.05 + RIGHT * (
                    1 - i) * 0.05
            point.scale(0.03)
            point.move_to(position)
            point.set_color(RED)
            points2b.append(point)

        self.play(
            *[FadeIn(_, run_time=0.5) for _ in points2b]
        )

        self.wait(7)

        # third case

        self.play(
            # ApplyMethod(newgroup1.move_to, shift1, run_time=2),
            # ApplyMethod(newgroup2.move_to, shift2, run_time=2),
            ApplyMethod(newgroup3.move_to, shift3, run_time=2)
        )

        arrows3 = []
        for i in range(0, 2):
            arrow = DashedVMobject(Arrow(DashedLine(UP * 0.1, UP * 0)))
            arrow.rotate(-np.pi / 2)
            #arrow.scale(0.5)
            position = arrows[i].get_center() + shift3 + DOWN * 2.05 + LEFT * 0.5
            arrow.move_to(position)
            arrow.set_color(WHITE)
            arrows3.append(arrow)

        points3 = []
        for i in range(0, 2):
            point = Circle(fill_color=WHITE, fill_opacity=1)
            position = points[i].get_center() + shift3 + DOWN * 2 + LEFT * 0.5
            point.scale(0.03)
            point.move_to(position)
            point.set_color(WHITE)
            points3.append(point)

        final_pos_arrows = (arrows3[0].get_center() + arrows3[1].get_center()) / 2
        final_pos_points = (points3[0].get_center() + points3[1].get_center()) / 2

        self.play(
            ApplyMethod(arrows3[0].move_to, final_pos_arrows, run_time=5),
            ApplyMethod(arrows3[1].move_to, final_pos_arrows, run_time=5),
            ApplyMethod(points3[0].move_to, final_pos_points, run_time=5),
            ApplyMethod(points3[1].move_to, final_pos_points, run_time=5)
        )

        self.wait(3)

        point3b = Circle(fill_color=RED, fill_opacity=1)
        point3b.scale(0.03)
        point3b.move_to(final_pos_points)
        point3b.set_color(RED)

        self.play(
            FadeIn(point3b, run_time=0.5)
        )

        self.wait(18)


class Example(Scene):
    def construct(self):
        title = TexText("Summary and Example", tex_to_color_map={"Summary": GREEN, "Example": YELLOW})
        title.move_to(UP * 3.75)

        entry1 = "Take any two elements of the list."
        entry2 = "Go through the list and find the next two elements,\\\\ that lie between the first two elements."
        entry3 = "Repeat the last step for all elements of the list."
        entry4 = "Show, that there is an element of the uncountably infinite set,\\\\ missing in the list."

        liste = BulletedList(entry1, entry2, entry3, entry4)
        liste.set_color(GREEN)

        self.play(
            Write(title)
        )

        self.wait(8)

        self.play(
            Write(liste, run_time=15)
        )
        self.wait(8)
        self.play(
            ApplyMethod(liste.scale_in_place, 0.7, run_time=1)
        )

        self.play(
            ApplyMethod(liste.move_to, 1.5 * UP + LEFT * 2, run_time=1)
        )

        rectangle = SurroundingRectangle(liste)
        rectangle.set_color(GREY)

        self.play(
            FadeIn(rectangle)
        )

        lift = 1

        positions = []
        l = 210
        for i in range(0, l):
            p = pos(i - 6.5, -1.5 + lift)
            positions.append(p)

        numbers = []
        for i in range(0, l):
            if i < 9:
                text = Tex("0", str(i + 1), tex_to_color_map={"0": BLACK})
                numbers.append(text)
            else:
                numbers.append(TexText(str(i + 1)))

        for i in range(0, l):
            numbers[i].move_to(RIGHT * positions[i].x + UP * positions[i].y)
            self.add(numbers[i])

        self.wait(1)

        boxes = []
        for i in range(0, l):
            box = SurroundingRectangle(numbers[i])
            box.set_color(YELLOW)
            box.stretch(2, 1)
            boxes.append(box)

        self.play(
            *[FadeIn(_, run_time=0.01) for _ in boxes],
            *[ApplyMethod(_.move_to, _.get_bottom() + DOWN * 0.75, run_time=0.01) for _ in boxes],
        )

        fractions = [Tex("0"), Tex("1")]
        fraction_vals = [0, 1]
        for i in range(2, 26):
            sublist = []
            for j in range(1, i + 1):
                if np.gcd(i, j) == 1:
                    fraction = Tex(r"\frac{" + str(j) + "}{" + str(i) + "}")
                    fractions.append(fraction)
                    fraction_vals.append(j / i)
                    fraction.scale(0.9)

        for i in range(0, len(fractions)):
            fractions[i].move_to(boxes[i].get_center())

        self.play(
            *[Write(_, run_time=0.1) for _ in fractions]
        )

        self.wait(3)

        number_line = NumberLine(x_min=0, x_max=10)
        number_line.move_to(DOWN * (4.5 - lift))

        blue = Color("blue")
        red = Color("red")

        colors = list(blue.range_to(red, l + 1))
        colors2 = list(red.range_to(blue, l + 1))

        color_line = Line(LEFT * 5, RIGHT * 5)
        color_line.set_color(colors2)
        color_line.move_to(DOWN * (4.5 - lift))

        self.play(
            FadeIn(number_line),
            FadeIn(color_line)
        )

        self.wait(3)

        removables = []
        arrows = []
        for i in range(7, 11):
            fraction = fractions[i].copy()
            removables.append(fraction)
            color = colors[math.floor(fraction_vals[i] * l)]
            fraction.set_color(color)
            fraction.scale(0.5)
            arrow = Arrow(Line(UP * 0.1, UP * 0))
            arrow.rotate(-np.pi / 2)
            arrow.scale(0.4)
            position = LEFT * 5 + 10 * fraction_vals[i] * RIGHT
            arrow.move_to(position + DOWN * (4.25 - lift))
            arrow.set_color(color)
            arrows.append(arrow)
            removables.append(arrow)
            self.play(
                ApplyMethod(fractions[i].set_color, color, run_time=0.01),
                ApplyMethod(fraction.move_to, position + DOWN * (3.75 - lift), run_time=0.5),
                FadeIn(arrow, run_time=0.5)
            )

        self.shifting(boxes, numbers, fractions, 30)

        self.wait(3)

        arrows = []
        for i in range(33, 43):
            fraction = fractions[i].copy()
            removables.append(fraction)
            color = colors[math.floor(fraction_vals[i] * l)]
            fraction.set_color(color)
            fraction.scale(0.5)
            arrow = Arrow(Line(UP * 0.1, UP * 0))
            arrow.rotate(-3. * np.pi / 2)
            arrow.scale(0.4)
            position = LEFT * 5 + 10 * fraction_vals[i] * RIGHT
            arrow.move_to(position + DOWN * (4.75 - lift))
            arrow.set_color(color)
            arrows.append(arrow)
            removables.append(arrow)
            self.play(
                ApplyMethod(fractions[i].set_color, color, run_time=0.01),
                ApplyMethod(fraction.move_to, position + DOWN * (4 - lift), run_time=0.25),
                FadeIn(arrow, run_time=0.25)
            )

        self.wait(3)

        self.play(
            *[FadeOut(rem) for rem in removables],
            *[ApplyMethod(frac.set_color, WHITE) for frac in fractions]
        )

        self.shifting(boxes, numbers, fractions, -30)

        self.next(1, 2, 0, fractions, fraction_vals, colors, [0, 0], boxes, numbers)
        self.next(4, 6, 1, fractions, fraction_vals, colors, [0, 0], boxes, numbers)
        self.next(17, 31, 2, fractions, fraction_vals, colors, [7, 14], boxes, numbers)
        self.next(92, 178, 3, fractions, fraction_vals, colors, [61, 86], boxes, numbers)

        self.wait(10)

    def shifting(self, boxes, numbers, fractions, shift):
        self.play(
            *[ApplyMethod(boxes[_].move_to, boxes[_].get_center() + LEFT * shift, run_time=shift / 10) for _ in
              range(0, len(boxes))],
            *[ApplyMethod(numbers[_].move_to, numbers[_].get_center() + LEFT * shift, run_time=shift / 10) for _ in
              range(0, len(numbers))],
            *[ApplyMethod(fractions[_].move_to, fractions[_].get_center() + LEFT * shift, run_time=shift / 10) for _ in
              range(0, len(fractions))]
        )

    def next(self, a, b, level, fractions, fraction_vals, colors, shifts, boxes, numbers):
        ints = [a, b]
        arrows = []
        lift = 1.5
        s = 0
        for i in ints:
            self.shifting(boxes, numbers, fractions, shifts[s])

            fraction = fractions[i].copy()
            color = colors[math.floor(fraction_vals[i] * (len(colors) - 1))]
            fraction.set_color(color)
            fraction.scale(0.5)
            arrow = Arrow(Line(UP * 0.1, UP * 0))
            arrow.rotate(-np.pi / 2)
            arrow.scale(0.4)
            position = LEFT * 5 + 10 * fraction_vals[i] * RIGHT
            arrow.move_to(position + DOWN * 3.25)
            arrow.set_color(color)
            arrows.append(arrow)

            left = Tex("[")
            left.scale(1.7)
            comma = Tex(",")
            right = Tex("]")
            right.scale(1.7)

            if fraction_vals[a] < fraction_vals[b]:
                frac_a = fractions[a].copy()
                frac_b = fractions[b].copy()
            else:
                frac_a = fractions[b].copy()
                frac_b = fractions[a].copy()

            frac_a.set_color(colors[math.floor(fraction_vals[a] * (len(colors) - 1))])
            frac_b.set_color(colors[math.floor(fraction_vals[b] * (len(colors) - 1))])

            left.move_to(UP * (3.5 - level) + RIGHT * 4)
            frac_a.move_to(UP * (3.5 - level) + RIGHT * 4.3)
            comma.move_to(UP * (3.5 - level) + RIGHT * 4.5)
            frac_b.move_to(UP * (3.5 - level) + RIGHT * 4.8)
            right.move_to(UP * (3.5 - level) + RIGHT * 5.1)

            s = s + 1

            if level < 3:
                self.play(
                    ApplyMethod(fractions[i].set_color, color, run_time=0.01),
                    ApplyMethod(fraction.move_to, position + DOWN * 2.75, run_time=1),
                    FadeIn(arrow, run_time=1)
                )
            else:
                self.play(
                    ApplyMethod(fractions[i].set_color, color, run_time=0.01),
                    FadeIn(arrow, run_time=1),
                    ApplyMethod(arrow.rotate, np.pi)
                )
                self.play(
                    ApplyMethod(arrow.move_to, arrow.get_center() + DOWN * 0.5)
                )
        self.play(
            Write(left),
            Write(frac_a),
            Write(comma),
            Write(frac_b),
            Write(right)
        )

        self.wait(3)


class Conclusion(Scene):
    def construct(self):
        title = TexText("Result")
        title.set_color(YELLOW)
        title.move_to(UP * 3.5)

        self.play(
            Write(title)
        )

        numbers = [1, 2, 1, 1, 2, 3, 3, 4, 7, 10, 5, 7, 12, 17, 17, 24, 41, 58, 29, 41, 70, 99, 99, 140, 239, 338, 169,
                   239]

        fractions = []
        for i in range(0, math.floor(len(numbers) / 2)):
            fraction = Tex(r"\frac{" + str(numbers[2 * i]) + "}{" + str(numbers[2 * i + 1]) + "}")
            if i > 10:
                fraction.stretch(0.7, 0)
            fractions.append(fraction)

        arrow = Tex(r"\rightarrow")

        start = LEFT * 7 + UP * 2
        offset = RIGHT * 1.6
        for i in range(0, 7):
            self.interval(fractions[2 * i], fractions[2 * i + 1], start + RIGHT * 2 * i)
            arr = arrow.copy()
            arr.move_to(start + offset + RIGHT * 2 * i)
            self.play(Write(arr))

        self.wait(1)

        observe = TexText("Observe:")
        observe.move_to(UP * 1)

        self.play(Write(observe))

        one = r"$\frac{3}{4} = \frac{3}{2\cdot 2}$"
        two = r"$\frac{7}{10} = \frac{7}{2\cdot 5}$"
        three = r"$\frac{17}{24} = \frac{17}{2\cdot 12}$"
        four = r"$\cdots$"

        liste = NoBulletedList(one, two, three, four)
        liste.move_to(DOWN * 1.5)
        liste.scale_in_place(0.9)

        self.play(
            Write(liste)
        )

        self.wait(3)
        self.play(
            FadeOut(observe),
            FadeOut(liste)
        )

        indices = [1, 2, 5, 6, 9, 10, 13]
        for i in indices:
            self.play(
                *[ApplyMethod(fractions[i].set_color, RED) for i in indices]
            )

        rule = Tex("\\cdots\\rightarrow\\Big[ a_n,\\frac{1}{2 \cdot a_n}\\Big]\\rightarrow",
                       tex_to_color_map={"a_n": RED})
        rule.move_to(LEFT * 2.2)
        rule2 = Tex("\\Big[\\frac{1}{2 \cdot a_{n+1}}, a_{n+1}\\Big]\\rightarrow\\cdots",
                        tex_to_color_map={"a_{n+1}": RED})
        rule2.move_to(RIGHT * 2.5)
        self.play(Write(rule))
        self.play(Write(rule2))
        rule3 = Tex("\\lim\\limits_{n\\rightarrow \\infty} a_n = \\frac{1}{\\sqrt{2}}",
                        tex_to_color_map={"a_n": RED, "\\frac{1}{\\sqrt{2}}": YELLOW})
        rule3.move_to(DOWN * 1.5)
        self.play(Write(rule3))
        rule4 = Tex(
            "\\lim\\limits_{n\\rightarrow \\infty} \\Big[ a_n,\\frac{1}{2 \cdot a_n}\\Big] = \Big[\\frac{1}{\\sqrt{2}},\\frac{1}{\\sqrt{2}}\\Big]",
            tex_to_color_map={"a_n": RED, "\\frac{1}{\\sqrt{2}}": YELLOW})
        rule4.move_to(DOWN * 3)
        self.play(Write(rule4))

        self.wait(3)

    def interval(self, fa, fb, pos):
        """ function that optimizes the creation of an interval

        Parameters
        ----------
        fa fraction a
        fb fraction b
        pos position on the screen

        Returns
        -------

        """
        left = Tex("[")
        left.scale(1.7)
        comma = Tex(",")
        right = Tex("]")
        right.scale(1.7)
        left.move_to(pos)
        fa.move_to(pos + RIGHT * 0.3)
        comma.move_to(pos + RIGHT * 0.6)
        fb.move_to(pos + RIGHT * 0.9)
        right.move_to(pos + RIGHT * 1.2)
        self.play(
            Write(left),
            Write(fa),
            Write(comma),
            Write(fb),
            Write(right)
        )


class ContinuedFractions(Scene):
    def construct(self):
        title = TexText(r"Continued Fractions for $\frac{1}{\sqrt{2}}$",
                    tex_to_color_map={r"$\frac{1}{\sqrt{2}}$": YELLOW, "Continued Fractions": RED})
        title.move_to(UP * 3.5)

        statement = [Tex(r"\frac{1}{44444444444444444}", tex_to_color_map={"44444444444444444": BLACK}),
                     Tex("="),
                     Tex(r"\frac{1}{444}", tex_to_color_map={"444": BLACK})]

        statement[0].shift(2 * LEFT)
        statement[1].next_to(statement[0], RIGHT)
        statement[2].next_to(statement[1], RIGHT)
        statement[2][0].set_color(YELLOW)

        den2 = Tex(r"\sqrt{2}")
        den2.set_color(YELLOW)
        den1 = Tex("1", "+", r"\frac{1}{2+\frac{1}{2+\frac{1}{2+\frac{1}{2+\cdots}}}}")

        den1.move_to(statement[0].get_center() + DOWN * 1)
        den2.move_to(statement[2].get_center() + DOWN * 0.35)

        den21 = Tex("x")
        den21.set_color(RED)
        den21.move_to(den2.get_center())

        self.play(
            Write(title)
        )

        self.play(
            Write(statement[0])
        )

        self.play(Write(den1))

        self.play(
            Write(statement[1])
        )
        self.play(
            Write(statement[2])
        )
        self.play(
            Write(den2)
        )

        self.wait(2)

        self.play(
            Transform(den2, den21)
        )

        self.wait(2)

        self.play(
            ApplyMethod(statement[0][0].set_color, BLUE),
            ApplyMethod(statement[2][0].set_color, BLUE)
        )

        self.wait(2)

        self.play(
            FadeOut(statement[0]),
            FadeOut(statement[2])
        )

        self.play(
            ApplyMethod(den1.move_to, statement[0].get_center() + DOWN * 0.3),
            ApplyMethod(den2.move_to, statement[2].get_center() + DOWN * 0)
        )

        neg = Tex("-")
        neg.next_to(den2, RIGHT)
        den1[0].generate_target()
        den1[0].target.next_to(neg, RIGHT)

        self.play(
            MoveToTarget(den1[0]),
            FadeOut(den1[1], run_time=0.0),
            Write(neg)
        )

        self.wait(2)

        den1[2].set_color(BLUE)
        den1[2][0:4].set_color(WHITE)

        self.wait(2)

        part = VGroup(den1[2][4:19])

        sub = Tex("x", "-1")
        sub[0].set_color(RED)
        sub.next_to(den1[2][3], RIGHT)

        self.play(
            Transform(part, sub)
        )
        self.wait(2)

        part2 = VGroup(den1[2][2:4], part)
        sub2 = Tex("x+1", tex_to_color_map={"x": RED})
        sub2.next_to(den1[2][1], DOWN)

        self.play(Transform(part2, sub2))
        self.wait(2)

        eq2 = Tex(r"(x-1)\cdot (x+1)", tex_to_color_map={"x": RED})
        rhs = VGroup(den2,neg,den1[0])
        eq2.next_to(statement[1],RIGHT)

        eq1 = Tex("1")
        eq1.next_to(statement[1],LEFT)
        lhs = VGroup(den1[2][0:2],part2)

        self.play(
            Transform(rhs, eq2),
            Transform(lhs,eq1)
        )
        self.wait(2)

        eq3 = Tex("1=x^2-1",tex_to_color_map={"x":RED})
        eq3.move_to(statement[1].get_center()+DOWN*0.92+RIGHT*0.58)

        self.play(
            FadeOut(eq2),
            Write(eq3)
        )

        self.wait(2)

        eq4 = Tex("2=x^2",tex_to_color_map={"x":RED})
        eq4.move_to(statement[1].get_center()+DOWN*1.8+RIGHT*0.1)

        self.play(
            Write(eq4)
        )

        self.wait(2)

        eq5 = Tex("\\sqrt{2}=x", tex_to_color_map={"=": WHITE,"x": RED})
        eq5.move_to(statement[1].get_center() + DOWN * 2.7+LEFT*0.2)

        self.play(
            Write(eq5)
        )

        self.wait(2)

        final = [Tex(r"\frac{1}{1+\frac{1}{2+\frac{1}{2+\frac{1}{2+\frac{1}{2+\cdots}}}}}"),
                Tex("="),
                Tex(r"\frac{1}{x}",tex_to_color_map={"x": WHITE})]

        final[0].move_to(LEFT*3+UP*2)
        final[1].next_to(final[0],RIGHT)
        final[2].next_to(final[1],RIGHT)

        final2= Tex(r"\sqrt{2}")
        final2.move_to(final[2].get_center())

        self.remove(*lhs)
        self.remove(*rhs)
        self.play(
            FadeOut(den1[2][0:2],den2,run_time=0.0),
            FadeOut(eq4, run_time=0.0),
            FadeOut(eq3, run_time=0.0),
            FadeOut(statement[1], run_time=0.0),
            FadeOut(sub, run_time=0.0),
            FadeOut(neg, run_time=0.0),
            FadeOut(den2, run_time=0.0),
            Write(final[0]),
            Write(final[1]),
            Write(final[2])
        )

        x = final[2][len(final[2])-2]
        self.wait(2)
        self.play(
            ApplyMethod(eq5[0].move_to,x.get_center(),run_time=0.5),
            FadeOut(x, run_time=1),
            FadeOut(eq5[1],run_time=1),
            FadeOut(eq5[2],run_time=1)
        )

        group = VGroup(*final,x,eq5[0])
        rectangle = SurroundingRectangle(group)
        rectangle.set_color(YELLOW)

        self.play(
            GrowFromCenter(rectangle)
        )

        self.wait(2)

        lStrings =["1","\\frac{2}{3}","\\frac{5}{7}","\\frac{12}{17}","\\frac{29}{41}",
                   "\\frac{70}{99}\\rule{1em}{0ex}","\\frac{169}{239}","\\frac{408}{577}"]
        rStrings=["\\frac{1}{1}",
                  "\\frac{1}{1+\\frac{1}{2}}",
                  "\\frac{1}{1+\\frac{1}{2+\\frac{1}{2}}}",
                  "\\frac{1}{1+\\frac{1}{2+\\frac{1}{2+\\frac{1}{2}}}}",
                  "\\frac{1}{1+\\frac{1}{2+\\frac{1}{2+\\frac{1}{2+\\frac{1}{2}}}}}",
                  "\\frac{1}{1+\\frac{1}{2+\\frac{1}{2+\\frac{1}{2+\\frac{1}{2+\\frac{1}{2}}}}}}",
                  "\\frac{1}{1+\\frac{1}{2+\\frac{1}{2+\\frac{1}{2+\\frac{1}{2+\\frac{1}{2+\\frac{1}{2}}}}}}}",
                  "\\frac{1}{1+\\frac{1}{2+\\frac{1}{2+\\frac{1}{2+\\frac{1}{2+\\frac{1}{2+\\frac{1}{2+\\frac{1}{2}}}}}}}}"
                  ]

        equal = []
        left = []
        right = []

        for i in range(0,8):
            equal.append(Tex("="))
            left.append(Tex(lStrings[i]))
            right.append(Tex(rStrings[i]))

        equal[0] = Tex("=")
        equal[0].move_to(UP*2.5+RIGHT)
        left[0].next_to(equal[0], LEFT)
        left[0].set_color(RED)
        right[0].next_to(equal[0], RIGHT)

        for i in range(1,8):
            equal[i] =Tex("=")
            if i < 5:
                equal[i].move_to(UP*(2.5-i*1.4)+RIGHT)
            else:
                equal[i].move_to(UP * (2.5 - (i-5) * 1.4) + RIGHT * 4.5)
            left[i].next_to(equal[i], LEFT)
            right[i].scale(np.power(0.9,i))
            right[i].next_to(equal[i], RIGHT)
            left[i].set_color(RED)
            if not i == 5:
                equal[i].align_to(equal[i-1],LEFT)
                left[i].align_to(left[i-1], RIGHT)
                right[i].align_to(right[i-1], LEFT)
            right[i].move_to(right[i].get_center() + DOWN * 0.05 * i)

        for i in range(0, 8):
            self.play(Write(left[i]))
            self.play(Write(equal[i]))
            self.play(Write(right[i]))

        self.wait(2)

        numbers = [1, 2, 1, 1, 2, 3, 3, 4, 7, 10, 5, 7, 12, 17, 17, 24, 41, 58, 29, 41, 70, 99, 99, 140, 239, 338, 169,
                   239,408,577,577,816]

        fractions = []
        for i in range(0, math.floor(len(numbers) / 2)):
            fraction = Tex(r"\frac{" + str(numbers[2 * i]) + "}{" + str(numbers[2 * i + 1]) + "}")
            if i > 10:
                fraction.stretch(0.7, 0)
            fractions.append(fraction)
            fraction.scale(0.7)

        arrow = Tex(r"\rightarrow")

        elems = []
        start = LEFT * 6.5+DOWN*1
        offset = RIGHT * 1.6
        shift = 0
        for i in range(0, 8):
            if i > 2:
                shift = 1
            if i > 5:
                shift = 2
            self.interval(fractions[2 * i], fractions[2 * i + 1], start + RIGHT * 2 * (i%3)+DOWN*shift,elems)
            arr = arrow.copy()
            if i<7:
                arr.move_to(start + offset + RIGHT * 2 * (i%3)+DOWN*shift)
                self.add(arr)
                elems.append(arr)

        group2 = VGroup(*elems)
        group2.align_to(group,RIGHT)
        rect2 = SurroundingRectangle(group2)
        rect2.set_color(GREY)
        self.play(GrowFromCenter(rect2))

        self.wait(1)

        indices = [1, 2, 5, 6, 9, 10, 13, 14]
        for i in indices:
            self.play(
                *[ApplyMethod(fractions[i].set_color, RED) for i in indices]
            )

        self.wait(10)

    def interval(self, fa, fb, pos, elems):
        left = Tex("[")
        left.scale(1.7)
        comma = Tex(",")
        right = Tex("]")
        right.scale(1.7)
        left.move_to(pos)
        delta = 0.6
        fa.move_to(pos + RIGHT * (0.3+delta))
        comma.move_to(pos + RIGHT * (0.6+delta))
        fb.move_to(pos + RIGHT * (0.9+delta))
        right.move_to(pos + RIGHT * (1.2))
        # fa.next_to(left,RIGHT,buff=4*SMALL_BUFF)
        # comma.next_to(fa,RIGHT,buff=SMALL_BUFF)
        # fb.next_to(comma,RIGHT,buff=SMALL_BUFF)
        # right.next_to(fb,RIGHT,buff=0)
        elems.append(left)
        elems.append(right)
        self.add(
            left,
            fa,
            comma,
            fb,
            right
        )


class TakeAway(Scene):
    def construct(self):
        title = TexText("Take-away")
        title.set_color(YELLOW)
        title.move_to(UP * 3.5)

        self.play(Write(title))
        self.wait(5)
        liste = BulletedList("countable sets: $\\mathbb{Z}$, $\\mathbb{Q}$, algebraic numbers",
                             "uncountable sets: $\\mathbb{R}$ due to transcendental numbers",
                             "continuum hypothesis: $\\aleph_1 = 2^{\\aleph_0}$", "incompleteness theorem")

        self.play(Write(liste[0]))
        self.wait(16)
        self.play(Write(liste[1]))
        self.wait(16)
        self.play(Write(liste[2]))
        self.play(Write(liste[3]))
        self.wait(14)
