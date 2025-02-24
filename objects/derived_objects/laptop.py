import numpy as np
from mathutils import Vector

from interface import ibpy
from objects.bobject import BObject
from utils.constants import OBJECT_APPEARANCE_TIME, FRAME_RATE, DEFAULT_ANIMATION_TIME
from utils.utils import flatten

dip = 0.025

layout = {
    'a': 24,
    'b': 42,
    'c': 40,
    'd': 26,
    'e': 14,
    'f': 27,
    'g': 28,
    'h': 29,
    'i': 19,
    'j': 30,
    'k': 31,
    'l': 32,
    'm': 44,
    'n': 43,
    'o': 20,
    'p': 21,
    'q': 12,
    'r': 15,
    's': 25,
    't': 16,
    'u': 18,
    'v': 41,
    'w': 13,
    'x': 39,
    'y': 17,
    'z': 38,
    'A': -24,
    'B': -42,
    'C': -40,
    'D': -26,
    'E': -14,
    'F': -27,
    'G': -28,
    'H': -29,
    'I': -19,
    'J': -30,
    'K': -31,
    'L': -32,
    'M': -44,
    'N': -43,
    'O': -20,
    'P': -21,
    'Q': -12,
    'R': -15,
    'S': -25,
    'T': -16,
    'U': -18,
    'V': -41,
    'W': -13,
    'X': -39,
    'Y': -17,
    'Z': -38,
    '0': 9,
    '1': 0,
    '2': 1,
    '3': 2,
    '4': 3,
    '5': 4,
    '6': 5,
    '^': -5,
    '7': 6,
    '8': 7,
    '9': 8,
    'f10': 68,
    'f11': 69,
    'f12': 70,
    'f1': 56,
    'f2': 60,
    'f3': 61,
    'f4': 62,
    'f5': 63,
    'f6': 64,
    'f7': 65,
    'f8': 66,
    'f9': 67,
    'shift1': 36,
    'shift2': 76,
    '\n': 77,
    ' ': 78,
    'ctrl1': 72,
    'ctrl2': 52,
    'left': 53,
    'right': 55,
    'up': 58,
    'down': 71,
    'alt1': 50,
    'alt2': 51,
    'esc': 57,
    'back': 75,
    'del': 59,
    '/': 47,
    '+': -11,
    '-': 10,
    '_':-10,
    '*': -7,
    '[': 22,
    ']': 23,
    '{': -22,
    '}': -23,
    '(': -8,
    ')': -9,
    ',': 45,
    '<':-45,
    '.': 46,
    '>':-46,
    '|': 37,
    ';': 33,
    ':':-33,
    '\'': 34,
    '\\': 35,
    '=': 11,
}

enter_interval = 0.5


class Laptop(BObject):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        location = self.get_from_kwargs('location', [0, 0, 0])
        rotation = self.get_from_kwargs('rotation_euler', [np.pi / 2, 0, 0])
        colors = self.get_from_kwargs('colors', flatten([['gray_3'] * 2, ['screen'], ['gray_5'] * 79, ['gray_1']]))
        name = self.get_from_kwargs('name', 'Laptop')

        key_strings = ["Key." + str(i).zfill(3) for i in range(79)]
        path_strings = ["path." + str(i).zfill(3) + "." + str(j).zfill(3) for i in range(79) for j in range(10)]

        bobs = BObject.from_file("Laptop", objects=["Display", "Keypad", "Screen", *key_strings, *path_strings],
                                 colors=colors, name=name)
        self.children = []
        self.keys = []
        self.key_labels = {}

        if len(bobs) > 0:
            self.display = bobs[0]
            self.children.append(self.display)
            if len(bobs) > 1:
                self.keypad = bobs[1]
                self.children.append(self.keypad)
            if len(bobs) > 1:
                self.screen = bobs[2]
                self.children.append(self.keypad)
                ibpy.set_parent(self.screen, self.display)
        for b in range(3, len(bobs)):
            bob = bobs[b]
            if 'Key' in bob.ref_obj.name:
                self.keys.append(bob)
                ibpy.set_parent(bob, self.keypad)
            if 'path' in bob.ref_obj.name:
                name = bob.ref_obj.name
                bob.set_emission_color('example')
                index = name.index('path')
                key_index = int(name[index + 5:index + 8])
                if key_index in self.key_labels:
                    self.key_labels[key_index].append(bob)
                else:
                    self.key_labels[key_index] = [bob]

                ibpy.set_parent(bob, self.keys[key_index])  # parent key layout to key
            self.children.append(bob)
        super().__init__(children=self.children, name=name, rotation_euler=rotation, location=location,
                         parenting=[0, 1])

    def appear(self,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME,
               **kwargs):
        super().appear(begin_time=begin_time, transition_time=transition_time)
        for child in self.children:
            offset=0
            if 'Key.' in child.ref_obj.name:
                offset=0.5
            child.appear(begin_time=begin_time+offset, transition_time=transition_time)
        self.screen.appear(begin_time=begin_time,transition_time=transition_time)
        return begin_time + transition_time

    def open(self, angle=90, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
        self.display.rotate(rotation_euler=[-angle / 180 * np.pi, 0, 0], pivot=[0, 0, -1.7], begin_time=begin_time,
                            transition_time=transition_time)
        return begin_time + transition_time

    def close(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        self.display.rotate(rotation_euler=[0,0,0],pivot=[0,0,-1.7],begin_time=begin_time,
                            transition_time=transition_time)
        return begin_time+transition_time

    def set_movie(self, src, duration, **kwargs):
        ibpy.set_movie_to_material(self.screen, src, duration=duration, **kwargs)

    def start_movie(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        begin_frame = begin_time * FRAME_RATE
        self.screen.change_emission(to_value=0.8)
        ibpy.set_movie_start(self.screen, begin_frame)
        # ibpy.mix_color(self.screen, from_value=0, to_value=1, begin_frame=begin_time * FRAME_RATE,
        #                frame_duration=transition_time * FRAME_RATE)
        return begin_time + transition_time

    def stop_movie(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        ibpy.mix_color(self.screen, from_value=1, to_value=0, begin_frame=begin_time * FRAME_RATE,
                       frame_duration=transition_time * FRAME_RATE)
        return begin_time + transition_time

    def key_press(self, key, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        t0 = begin_time
        addon = 0

        if key == 77:
            addon = enter_interval

        keys = []
        if key < 0:
            keys.append(layout['shift1'])
            key = np.abs(key)
        keys.append(key)

        for key in keys:
            self.keys[key].move(direction=[0, -dip, 0], begin_time=t0, transition_time=transition_time / 2)
            self.keys[key].move_to(target_location=Vector(), begin_time=t0 + transition_time - transition_time / 3,
                                transition_time=transition_time / 3)
            for label in self.key_labels[key]:
                label.change_emission(from_value=0, to_value=10, begin_time=t0, transition_time=transition_time / 2)
                label.change_emission(from_value=10, to_value=0, begin_time=t0 + transition_time - transition_time / 3,
                                      transition_time=transition_time / 3)
        return t0 + transition_time+addon

    def write(self, text, begin_time=0, interval=DEFAULT_ANIMATION_TIME, **kwargs):
        print("writing started at ",text, begin_time *FRAME_RATE)
        if 'transition_time' in kwargs:
            transition_time = kwargs.pop('transition_time')
            interval = transition_time / len(text)
        t0 = begin_time
        for c in text:
            key = layout[c]
            t0 = self.key_press(key, begin_time=t0, transition_time=interval)
        return t0


    def escape(self,text,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        print("esc started at ", begin_time *FRAME_RATE)
        t0 = begin_time
        t0 = self.key_press(layout['esc'], begin_time=t0, transition_time=1/3)
        t0 = self.write(text,begin_time=t0,interval=transition_time-1/2)
        return self.key_press(layout['esc'], begin_time=t0, transition_time=1/6)

    def heading(self,text,begin_time=0,interval=DEFAULT_ANIMATION_TIME,level=5):
        t0 = begin_time
        print("heading started at ",t0*FRAME_RATE)
        # switch to comment mode
        toggles = [
            layout['alt1'],
            layout[str(level)]
        ]
        for t in toggles:
            self.key_press(t,begin_time=t0,transition_time=1/3)
        t0+=1/3

        for c in text:
            key = layout[c]
            t0=self.key_press(key,begin_time=t0,transition_time=interval)

        #leave heading
        return self.down(begin_time=t0,transition_time=0.5)

    def comment(self, text, begin_time=0, interval=DEFAULT_ANIMATION_TIME):
        t0 = begin_time
        print("comment started at ", t0 *FRAME_RATE)
        # switch to comment mode
        toggles = [
            layout['alt1'],
            layout['6']
        ]
        for t in toggles:
            self.key_press(t, begin_time=t0, transition_time=interval)
        t0 += interval

        for c in text:
            key = layout[c]
            t0 = self.key_press(key, begin_time=t0, transition_time=interval)

        # leave comment mode
        t0 = self.down(begin_time=t0, transition_time=1 / 3)
        return t0

    def down(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        down = layout['down']
        self.key_press(down, begin_time=begin_time, transition_time=transition_time)
        print("paragraph left at ", (begin_time+transition_time) *FRAME_RATE)
        return begin_time + transition_time

    def sub(self, begin_time = 0, transition_time = DEFAULT_ANIMATION_TIME):
        print("sub started at ",begin_time *FRAME_RATE)
        combine = [
            layout['ctrl1'],
            layout['-']
        ]
        for c in combine:
            self.key_press(c, begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def subtext(self,text,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        t0=begin_time
        t0=self.sub(begin_time=t0,transition_time=1/4)
        return 1/4+self.write(text,begin_time=t0,interval=np.maximum(0,(transition_time-1/2)/len(text)))


    def suptext(self,text,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        t0=begin_time
        t0=self.sup(begin_time=t0,transition_time=1/4)
        return 1/4+self.write(text,begin_time=t0,interval=np.maximum(0,(transition_time-1/2)/len(text)))

    def divide(self,text,begin_time=0,interval=DEFAULT_ANIMATION_TIME):
        t0=begin_time
        combine=[
            layout['ctrl1'],
            layout['/']
        ]
        for c in combine:
            self.key_press(c,begin_time=t0,transition_time=1/3)

        t0+=1/3
        t0=self.write(text,begin_time=t0,interval=interval)
        return self.right(begin_time=t0,transition_time=1/3)

    def sup(self, begin_time, transition_time = DEFAULT_ANIMATION_TIME):
        print("super started at ", begin_time*FRAME_RATE)
        combine = [
            layout['ctrl1'],
            layout['^']
        ]
        for c in combine:
            self.key_press(c, begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def sup2(self, text,begin_time: object = 0, interval = DEFAULT_ANIMATION_TIME) -> object:
        t0=begin_time
        combine = [
            layout['ctrl1'],
            layout['^']
        ]
        for c in combine:
            self.key_press(c, begin_time=t0, transition_time=1/3)
        t0+=1/3
        t0=self.write(text,begin_time=t0,interval=interval)
        return self.right(begin_time=t0,transition_time=1/3)

    def right(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.key_press(layout['right'], begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def enter(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        print("enter started at ",begin_time *FRAME_RATE)
        self.key_press(layout['\n'], begin_time=begin_time, transition_time=transition_time/2)
        return begin_time+transition_time

    def evaluate(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        print("evaluate started at ", begin_time *FRAME_RATE)
        combine = [
            layout['shift1'],
            layout['\n']
        ]
        for c in combine:
            self.key_press(c, begin_time=begin_time, transition_time=transition_time)

        return begin_time + transition_time + enter_interval
