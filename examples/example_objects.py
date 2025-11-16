import os
from collections import OrderedDict

import numpy as np

from interface import ibpy
from interface.ibpy import Vector, Quaternion
from interface.interface_constants import BLENDER_EEVEE, CYCLES
from objects.empties import EmptyCube
from objects.geometry.sphere import Sphere, MultiSphere, invert, HalfSphere, StackOfSpheres
from perform.scene import Scene
from utils.constants import DATA_DIR
from utils.utils import print_time_report
from utils.utils_io import parse

pi = np.pi
tau = 2*pi

def read_data(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path) as f:
        contents = f.read()

    str_data = contents.split(',')
    data = [parse(dat) for dat in str_data]
    return data


class ExamplesObjects(Scene):
    """
    This scene contains examples for complex animations for reference
    """
    def __init__(self):
        self.t0 = None
        self.construction_counter = 0
        self.old = None
        self.sub_scenes = OrderedDict([
            ('sphere', {'duration': 10}),
            ('half_sphere', {'duration': 10}),
            ('multi_sphere', {'duration': 10}),
            ('stack_of_sphere', {'duration': 40}),
        ])
        super().__init__(light_energy=2, transparent=False)

    def sphere(self):
        t0 = 0

        ibpy.set_hdri_background("qwantani_puresky_4k", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=False, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=512,
                               motion_blur=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [15,-15,5]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=140)



        sphere = Sphere(location=[0, 0, 0], radius=10,color="gold",smooth=3)
        t0 = 0.5 + sphere.grow(begin_time=t0, transition_time=1)

        self.t0 = t0

    def multi_sphere(self):
        t0 = 0

        ibpy.set_hdri_background("qwantani_puresky_4k", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=False, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=512,
                               motion_blur=False,shadows = False)

        camera_empty = EmptyCube(location=Vector([0, -0.5, 0]))
        camera_location = [0,-5,5]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=140)

        # show spheres
        all_points = []
        all_curvatures = []
        max_curvature = 100
        # create spheres up to a given resolution
        N = 10
        points = read_data("centers.dat")
        curvatures = read_data("curvatures.dat")

        for center, curvature in zip(points, curvatures):
            for i in set(range(0, N + 1)) | set(range(-N, 0)):
                M, R = invert(center + i, 1 / curvature)
                if curvature < max_curvature and 1 / R < max_curvature:
                    all_points.append(center + i)  # translate to different sectors of the band
                    all_curvatures.append(curvature)
                    if np.imag(center) != 0.5:
                        all_points.append(np.real(center) + i + 1j - np.imag(center) * 1j)  # mirror along the half line
                        all_curvatures.append(curvature)

        multi_sphere = MultiSphere(all_points, all_curvatures, name="ApollonianPackage", mesh_type='ico',
                                   color='vertex_color',attribute="curvature_color", max_curvature=max_curvature)

        t0 = 0.5 + multi_sphere.appear(begin_time=t0, transition_time=1)

        self.t0 = t0

    def half_sphere(self):
        t0 = 0

        ibpy.set_hdri_background("qwantani_puresky_4k", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * Vector([0,-157,-283]))
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=False, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=512,
                               motion_blur=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [18.7,-15,0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=140)

        half_sphere = HalfSphere(1 / 2, location=[0, 0, 1 / 2], resolution=100, solid=0.01, offset=0,color="example",roughness=0,metallic=0.5)
        t0 = half_sphere.appear(begin_time=t0)

        self.t0 = t0

    def stack_of_sphere(self):
        t0 = 0

        ibpy.set_hdri_background("qwantani_puresky_4k", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * Vector([0,-157,-283]))
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=False, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=512,
                               motion_blur=False)

        camera_empty = EmptyCube(location=Vector([0, 6,4]))
        camera_location = [18.7,-15,6.5]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=30)

        sequence = [50, 20, 20, 20, 3, 1, 1]
        stack_b = StackOfSpheres(radius=0.5, number_of_spheres=100, color='important', smooth=2,
                             location=[2.4894, 2.8364, 0.116], name="StackBob", scale=2)
        duration=30
        dt = 8 * duration / 10 / len(sequence)
        for i in range(len(sequence)):
            stack_b.appear(incr=sequence[i], begin_time=t0 + dt / 3, transition_time=dt / 3)
            t0 += dt

        self.t0 = t0

if __name__ == '__main__':
    try:
        example = ExamplesObjects()
        dictionary = {}
        for i, scene in enumerate(example.sub_scenes):
            print(i, scene)
            dictionary[i] = scene
        if len(dictionary) == 1:
            selected_scene = dictionary[0]
        else:
            choice = input("Choose scene:")
            if len(choice) == 0:
                choice = 0
            print("Your choice: ", choice)
            selected_scene = dictionary[int(choice)]

        example.create(name=selected_scene, resolution=[1920, 1080], start_at_zero=True)

        # example.render(debug=True)
        # doesn't work
        # example.final_render(name=selected_scene,debug=False)
    except:
        print_time_report()
        raise ()
