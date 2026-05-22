"""
Extra example scenes that showcase individual objects from ``objects/``.

Companion to ``example_objects.py`` (which covers the Sphere family). These
scenes are deliberately *self-contained*: they do not depend on external
HDRI maps or data files, so they build and render on any machine. Each
sub-scene sets up a small studio (camera + fill light on top of the default
sun that ``initialize_blender`` provides), creates one object with a colour,
and animates it in.

Run a single scene (interactive picker):
    python example_more_objects.py
Or headless via the documentation runner (see examples/object_gallery).
"""
from collections import OrderedDict

import numpy as np

from interface import ibpy
from interface.ibpy import Vector
from interface.interface_constants import CYCLES
from objects.empties import EmptyCube
from objects.cube import Cube
from objects.cylinder import Cylinder
from objects.cone import Cone
from objects.torus import Torus
from objects.geometry.sphere import Sphere
from objects.arrow import Arrow
from perform.scene import Scene
from utils.utils import print_time_report

pi = np.pi
tau = 2 * pi


class ExamplesMoreObjects(Scene):
    """Reference scenes for the basic geometric objects."""

    def __init__(self):
        self.t0 = None
        self.sub_scenes = OrderedDict([
            ('cube', {'duration': 5}),
            ('cylinder', {'duration': 5}),
            ('cone', {'duration': 5}),
            ('torus', {'duration': 5}),
            ('sphere_trio', {'duration': 5}),
            ('arrow', {'duration': 5}),
        ])
        # a sun is added by initialize_blender (LIGHT_TYPE='SUN')
        super().__init__(light_energy=3, transparent=False)

    # ---- shared studio setup -------------------------------------------
    def _studio(self, cam_loc, target=Vector(), lens=50, fill=(5, -5, 7),
                fill_energy=700):
        ibpy.set_render_engine(denoising=False, transparent=False,
                               frame_start=1, resolution_percentage=100,
                               engine=CYCLES, taa_render_samples=64,
                               motion_blur=False)
        camera_empty = EmptyCube(location=to_vec(target))
        ibpy.set_camera_location(location=cam_loc)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=lens)
        # warm key-fill point light in addition to the default sun
        ibpy.add_point_light(location=list(fill), energy=fill_energy)

    # ---- scenes --------------------------------------------------------
    def cube(self):
        t0 = 0
        self._studio(cam_loc=[6, -6, 4.5], lens=55)
        cube = Cube(location=[0, 0, 0], color="drawing", name="DemoCube")
        t0 = 0.5 + cube.grow(begin_time=t0, transition_time=1)
        self.t0 = t0

    def cylinder(self):
        t0 = 0
        self._studio(cam_loc=[6, -6, 4.5], lens=55)
        cyl = Cylinder(location=[0, 0, 0], length=2.2, radius=0.8,
                       color="example", name="DemoCylinder")
        t0 = 0.5 + cyl.grow(begin_time=t0, transition_time=1)
        self.t0 = t0

    def cone(self):
        t0 = 0
        self._studio(cam_loc=[6, -6, 4.5], lens=55)
        cone = Cone(location=[0, 0, -1], length=2.5, radius=1.0,
                    color="important", name="DemoCone")
        cone.grow(begin_time=t0, transition_time=1)   # Cone.grow returns None
        self.t0 = t0 + 1.5

    def torus(self):
        t0 = 0
        self._studio(cam_loc=[0, -7, 5], lens=55)
        torus = Torus(location=[0, 0, 0], rotation_euler=(0.4, 0, 0),
                      major_segments=96, minor_segments=24,
                      major_radius=1.6, minor_radius=0.45,
                      color="joker", name="DemoTorus")
        t0 = 0.5 + torus.appear(begin_time=t0, transition_time=1)
        self.t0 = t0

    def sphere_trio(self):
        t0 = 0
        self._studio(cam_loc=[0, -8, 3], target=Vector([0, 0, 0]), lens=55)
        cols = ["drawing", "example", "important"]
        for i, (x, c) in enumerate(zip([-2.2, 0, 2.2], cols)):
            s = Sphere(location=[x, 0, 0], radius=0.9, color=c, smooth=3,
                       name="Sphere_%d" % i)
            t0 = s.grow(begin_time=i * 0.3, transition_time=1)
        self.t0 = 1.5

    def arrow(self):
        t0 = 0
        self._studio(cam_loc=[5, -6, 1.5], target=Vector([0, 0, 0.2]),
                     lens=45)
        arrow = Arrow.from_start_to_end(start=[0, 0, -1.6], end=[0, 0, 1.8],
                                        radius=0.12, color="important",
                                        name="DemoArrow")
        arrow.grow(begin_time=t0, transition_time=1.2)   # may return None
        self.t0 = t0 + 1.7


def to_vec(v):
    return v if isinstance(v, Vector) else Vector(v)


if __name__ == '__main__':
    try:
        example = ExamplesMoreObjects()
        dictionary = {}
        for i, scene in enumerate(example.sub_scenes):
            print(i, scene)
            dictionary[i] = scene
        choice = input("Choose scene:")
        if len(choice) == 0:
            choice = 0
        selected_scene = dictionary[int(choice)]
        example.create(name=selected_scene, resolution=[1920, 1080],
                       start_at_zero=True)
    except Exception:
        print_time_report()
        raise
