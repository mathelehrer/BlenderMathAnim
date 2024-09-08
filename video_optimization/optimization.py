import random
from collections import OrderedDict

import numpy as np

from geometry_nodes.geometry_nodes import de_bruijn, penrose_3D_analog, create_node_for_optimization
from interface import ibpy
from interface.ibpy import Vector, get_geometry_node_from_modifier, change_default_value, \
    set_material, make_rigid_body
from interface.interface_constants import BLENDER_EEVEE
from mathematics.geometry.convex_hull import ConvexHull
from mathematics.mathematica.mathematica import tuples
from objects.circle import BezierCircle
from objects.coordinate_system import CoordinateSystem
from objects.cube import Cube
from objects.cylinder import Cylinder
from objects.derived_objects.p_arrow import PArrow
from objects.empties import EmptyCube
from objects.geometry.sphere import Sphere
from objects.image import ReferenceImage
from objects.plane import Plane
from objects.polygon import Polygon
from objects.table import Table
from objects.tex_bobject import SimpleTexBObject
from perform.scene import Scene
from utils.utils import print_time_report, pi, flatten

pi = np.pi


class Optimization(Scene):
    def __init__(self):
        self.t0 = None
        self.construction_counter = 0
        self.old = None
        self.sub_scenes = OrderedDict([
            ('intro', {'duration': 20}),
        ])
        super().__init__(light_energy=2, transparent=False)

    def intro(self):
        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 no_transmission_ray='True',
                                 rotation_euler=[0, 0, 260 * pi / 180])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine=BLENDER_EEVEE,
                               taa_render_samples=64  # 1024 for good alpha channel in EEVEE
                               )

        t0 = ibpy.set_hdri_strength(1, transition_time=0.5)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        # add any object, its geometry is not important, it will be replaced by the geometry node network
        function_plot = Cube()
        function_nodes = create_node_for_optimization()
        function_plot.add_mesh_modifier(type='NODES',node_group=function_nodes)

        self.t0 = t0+5


if __name__ == '__main__':
    try:
        example = Optimization()
        dictionary = {}
        for i, scene in enumerate(example.sub_scenes):
            print(i, scene)
            dictionary[i] = scene

        if len(dictionary)==1:
            selected_scene=dictionary[0]
        else:
            choice = input("Choose scene:")
            if len(choice)==0:
                choice=0
            print("Your choice: ", choice)
            selected_scene = dictionary[int(choice)]


        example.create(name=selected_scene, resolution=[1920, 1080], start_at_zero=True)

        # example.render(debug=True)
        # doesn't work
        # example.final_render(name=selected_scene,debug=False)
    except:
        print_time_report()
        raise ()
