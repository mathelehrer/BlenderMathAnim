import collections

import numpy as np

from interface import ibpy
from objects.coordinate_system import CoordinateSystem
from objects.tex_bobject import SimpleTexBObject
from perform.render import render_with_skips
from perform.scene import Scene
from utils.utils import print_time_report


class ExampleScene(Scene):
    def __init__(self):
        self.sub_scenes=collections.OrderedDict([
            ('intro',{'duration':14}),
            ('preview',{'duration':20}),
            ('main',{'duration':60})
        ])
        # in the super constructor the timing is set for all scenes
        super().__init__()

    def play(self):
        super().play()

        self.intro()
        self.preview()
        self.main()

    def intro(self):
        cues = self.sub_scenes['intro']
        title = SimpleTexBObject("Into",color='important')
        title.appear(begin_time=cues['start'] + 0, writing_time=10)

        labels = []
        for i in range(11):
            labels.append(SimpleTexBObject(str(i)))

        for i,label in enumerate(labels):
            label.appear(begin_time=t0 + i, duration=1)

        to_disappear=[]
        for i,things in enumerate(to_disappear):
            things.dissappear(begin_time=cues['end']-(len(to_disappear)-1-i)*0.05)

    def preview(self):
        cues = self.sub_scenes['preview']
        title = SimpleTexBObject("preview")
        title.appear(begin_time=cues['start'] + 0, writing_time=10)

        to_disappear = []
        for i, things in enumerate(to_disappear):
            things.dissappear(begin_time=cues['end'] - (len(to_disappear) - 1 - i) * 0.05)

    def main(self):
        cues = self.sub_scenes['main']
        title = SimpleTexBObject("main")
        title.appear(begin_time=cues['start'] + 0, writing_time=80)

        to_disappear = []
        for i, things in enumerate(to_disappear):
            things.dissappear(begin_time=cues['end'] - (len(to_disappear) - 1 - i) * 0.05)


if __name__ == '__main__':
    try:
        example = ExampleScene()
        example.play()
        ibpy.save(type(example).__name__)
        start = ibpy.start_frame()
        end = ibpy.end_frame()
        render_with_skips(start,end)
    except:
        print_time_report()
        raise()