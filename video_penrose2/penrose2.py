import random
from collections import OrderedDict

import numpy as np

from geometry_nodes.geometry_nodes import de_bruijn, penrose_3D_analog
from interface import ibpy
from interface.ibpy import Vector, get_geometry_node_from_modifier, change_default_value, \
    set_material, make_rigid_body
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


class BTable:
    pass


class Penrose(Scene):
    def __init__(self):
        self.t0 = None
        self.construction_counter = 0
        self.old = None
        self.sub_scenes = OrderedDict([
            ('z5_ternary', {'duration': 20}),
            ('convex_hull', {'duration': 20}),
            ('face_annotation', {'duration': 10}),
            ('face_selection', {'duration': 36}),
            ('binaries', {'duration': 10}),
            ('projection_equations', {'duration': 3}),
            ('title', {'duration': 3}),
            ('projections', {'duration': 7}),
            ('z3_annotations', {'duration': 5}),
            ('z3_ternary', {'duration': 15}),
            ('z3_example', {'duration': 5}),
            ('z3_lattice', {'duration': 10}),
            ('z3', {'duration': 10}),
            ('ten_rotations', {'duration': 3}),
            ('basis_rotation', {'duration': 7}),
            ('matrix_multiplication', {'duration': 9}),
            ('basis', {'duration': 9}),
            ('rotations', {'duration': 11}),
            ('the_wall', {'duration': 15}),
            ('the_end_of_the_wall', {'duration': 15}),
        ])
        super().__init__(light_energy=2, transparent=False)

    def z5_ternary(self):
        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 no_transmission_ray='True',
                                 rotation_euler=[0, 0, 260 * pi / 180])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine="BLENDER_EEVEE",
                               taa_render_samples=1024  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0.5)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        examples = {0: "000_3",
                    1: "001_3",
                    2: "002_3",
                    3: "010_3",
                    6: "020_3",
                    9: "100_3",
                    10: "101_3",
                    26: "222_3"}

        examples5 = {0:"00000_3",
                    1:"00001_3",
                    2:"00002_3",
                    3:"00010_3",
                    6:"00020_3",
                    9:"00100_3",
                    10:"00101_3",
                    26:"00222_3",
                    242:"22222_3"}

        count = 0
        for key,val in examples.items():
            text = SimpleTexBObject(str(key)+r"\mapsto"+val+"\mapsto"+examples5[key],location = [-0.5,0,3-0.4*count])
            text.write(begin_time=t0+0.5*count,transition_time=0.5)
            count+=1
        text = SimpleTexBObject(str(242) + r"\hspace{3em}\mapsto" + examples5[242], location=[-0.5, 0, 3 - 0.4 * count])
        text.write(begin_time=t0 + 0.5 * count, transition_time=0.5)
        count += 1
        self.t0 = t0+5
    def convex_hull(self):
        """
        rays and convex hull
        :return:
        """
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 no_transmission_ray='True',
                                 rotation_euler=[0, 0, 260 * pi / 180])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine="BLENDER_EEVEE",
                               taa_render_samples=1024  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0.5)

        empty = EmptyCube(location=[0, 0, 1])

        camera_circle = BezierCircle(location=[0, 0, 1], radius=5)

        ibpy.set_camera_view_to(empty)
        ibpy.set_camera_follow(camera_circle)
        ibpy.set_camera_location(location=[0,0,0])

        # define plane and orthogonal space
        r5 = np.sqrt(5)
        r2 = np.sqrt(2)
        u1 = -0.5 * np.sqrt(3 / 5 + 1 / r5)
        u2 = (-1 + r5) / 2 / r2 / r5
        v1 = 0.5 * np.sqrt(1 - 1 / r5)
        v2 = -0.5 * np.sqrt(1 + 1 / r5)

        ortho3 = np.array([[u2, u1, u1, u2, r2 / r5],
                           [-v2, v1, -v1, v2, 0],
                           [1 / r5, 1 / r5, 1 / r5, 1 / r5, 1 / r5]])

        #visualizing the complex hull
        voronoiCell = tuples([0,1],5)

        cell_points = [np.dot(ortho3, Vector(v)) for v in voronoiCell]
        ch = ConvexHull(cell_points)

        spheres = [Sphere(0.025, location=point, color='plastic_joker') for point in ch.points]
        for sphere in spheres:
            sphere.grow(begin_time=t0, transition_time=0.5)
            t0 += 0.1

        faces = [Polygon(vertices=[Vector(v) for v in ch.points[list(indices)]], color='joker',transmission=0.5) for indices
                 in ch.faces.keys()]

        for face in faces:
            face.appear(begin_time=t0, transition_time=0.5,alpha=0.125)
            t0 += 0.1

        #test intersection with a face
        #each face of the convex hull is given by a triple of vertices (triangulated convex hull)

        location = [-0.1, 0.1, 0.75]
        point = Sphere(0.05, location =location, mesh="iso", resolution=2, smooth=0,
                       name="SamplePoint", color="example")
        t0 = 0.5+ point.grow(begin_time=t0,transition_time=0.5)

        result = ch.ray_cast(location,direction=Vector([0,0,-1]))

        ts = result.keys()
        for t in ts:
            res = result[t]
            ray = Cylinder.from_start_to_end(start=location,end=res[1],radius=0.01,color='text',emission=10)
            t1 = ray.grow(begin_time=t0,transition_time=5*np.abs(t))
            idx = list(ch.faces.keys()).index(res[0])
            faces[idx].disappear(alpha=1,begin_time=t1,transition_time=0)

        t0 = 20

        ibpy.camera_follow(camera_circle,initial_value=0,final_value=0.5,begin_time=0,transition_time=t0)
        self.t0 = t0

    def rotation_5d(self):

        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 no_transmission_ray='True',
                                 rotation_euler=[0, 0, 260 * pi / 180])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine="BLENDER_EEVEE",
                               taa_render_samples=128  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0.5)

        ref = ReferenceImage("ref5.png", scale=2.641, rotation_euler=[pi / 2, 0, 0])
        ref.appear(begin_time=t0, transition_time=0)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        line = SimpleTexBObject(
            r"\left(\begin{array}{c c c c c} 1&0&0&0&0\\0&1&0&0&0\\0&0&1&0&0\\0&0&0&\cos\vartheta_9&-\sin\vartheta_9\\0&0&0&\sin\vartheta_9&\cos\vartheta_9\end{array}\right)=\left(\begin{array}{c c} {\rm row}_{00} & {\rm row}_{01}\\{\rm row}_{10} & {\rm row}_{11}\\ {\rm row}_{20} & {\rm row}_{21} \\ {\rm row}_{30} & {\rm row}_{31} \\{\rm row}_{40} & {\rm row}_{41}\end{array}\right)",
            text_size="small", location=[0.8, 0, -0.75], aligned="left")
        line2 = SimpleTexBObject(
            r"\left(\begin{array}{c c c c c c} 1&0&0&0&0&0\\0&1&0&0&0&0\\0&0&1&0&0&0\\0&0&0&\cos\vartheta_9&-\sin\vartheta_9&0\\0&0&0&\sin\vartheta_9&\cos\vartheta_9&0\end{array}\right)=\left(\begin{array}{c c} {\rm row}_{00} & {\rm row}_{01}\\{\rm row}_{10} & {\rm row}_{11}\\ {\rm row}_{20} & {\rm row}_{21} \\ {\rm row}_{30} & {\rm row}_{31} \\{\rm row}_{40} & {\rm row}_{41}\end{array}\right)",
            text_size="small", location=[0.8, 0, -0.75], aligned="left")
        t0 = 0.25 + line.write(
            letter_set=[4, 0, 1, 2, 3, 5, 48, 50, 51, 52, 53, 49, 6, 11, 16, 27, 40, 7, 12, 17, 28, 41, 8, 13, 18, 29,
                        42, 9, 14, 19, 21, 24, 26, 30, 32, 34, 36, 38, 43, 45, 47, 10, 15, 20, 22, 23, 25, 31, 33, 35,
                        37, 39, 44, 46], begin_time=t0, transition_time=2)

        movers = [48, 50, 51, 52, 53, 49]
        for mover in movers:
            line.letters[mover].move(direction=[0.4, 0, 0], begin_time=t0, transition_time=0.25)
        t0 = 0.5 + t0

        t0 = 0.25 + line2.write(letter_set=[48, 49, 50, 51, 52], begin_time=t0, transition_time=0.25)

        old_set = [54, 59, 55, 56, 57, 58, 60, 111, 113, 114, 115, 116, 112, 61, 66, 71, 76, 81, 86, 91, 96, 101, 106,
                   62, 67, 72, 77, 82, 87, 92, 97, 102, 107, 63, 68, 73, 78, 83, 88, 93, 98, 103, 108, 64, 69, 74, 79,
                   84, 89, 94, 99, 104, 109, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110]
        transformed_set = [o + 5 for o in old_set]
        t0 = 0.5 + line2.write(letter_set=transformed_set, begin_time=t0, transition_time=1.2)
        self.t0 = t0

    def basis_5d(self):

        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 no_transmission_ray='True',
                                 rotation_euler=[0, 0, 260 * pi / 180])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine="BLENDER_EEVEE",
                               taa_render_samples=128  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0.5)

        ref = ReferenceImage("ref4.png", scale=2.641, rotation_euler=[pi / 2, 0, 0])
        ref.appear(begin_time=t0, transition_time=0)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        lines = [
            SimpleTexBObject(r"\vec{u}=\left(\begin{array}{c}1\\0\\0\\0\\0\end{array}\right)", aligned='right',
                             color='custom1', location=[-0.22, 0, 2.25], text_size='tiny'),
            SimpleTexBObject(r"\vec{v}=\left(\begin{array}{c}0\\1\\0\\0\\0\end{array}\right)", aligned='right',
                             color='custom1', location=[-0.22, 0, 1.05], text_size='tiny'),
            SimpleTexBObject(r"\vec{w}_1=\left(\begin{array}{c}0\\0\\1\\0\\0\end{array}\right)", aligned='right',
                             color='drawing', location=[-0.22, 0, -0.05], text_size='tiny'),
            SimpleTexBObject(r"\vec{w}_1=\left(\begin{array}{c}0\\0\\0\\1\\0\end{array}\right)", aligned='right',
                             color='drawing', location=[-0.22, 0, -1.16], text_size='tiny'),
            SimpleTexBObject(r"\vec{w}_2=\left(\begin{array}{c}0\\0\\0\\0\\1\end{array}\right)", aligned='right',
                             color='drawing', location=[-0.22, 0, -2.34], text_size='tiny'),
        ]
        count = 0
        for line in lines:
            if count < 2:
                t0 = 0.5 + line.write(letter_set=[0, 1, 2, 7, 3, 4, 5, 6, 8, 14, 16, 17, 18, 19, 15, 9, 10, 11, 12, 13],
                                      begin_time=t0, transition_time=0.5)
            else:
                t0 = 0.5 + line.write(
                    letter_set=[0, 1, 2, 3, 8, 4, 5, 6, 7, 9, 15, 17, 18, 19, 20, 16, 10, 11, 12, 13, 14],
                    begin_time=t0, transition_time=0.5)
            count += 1

        self.t0 = t0

    def face_annotation(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 no_transmission_ray='True',
                                 rotation_euler=[0, 0, 260 * pi / 180])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine="BLENDER_EEVEE",
                               taa_render_samples=128  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0.5)

        ref = ReferenceImage("ref3.png", scale=2.641, rotation_euler=[pi / 2, 0, 0])
        ref.appear(begin_time=t0, transition_time=0)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        lines = [
            SimpleTexBObject(r"\text{Corner computation}",location=[-5.5,0,-1],text_size='tiny'),
            SimpleTexBObject(r"\text{and projection}",location=[-5.5,0,-1.3],text_size='tiny'),
            SimpleTexBObject(r"\text{Convex hull test}", location=[-3.1, 0, -0.5], text_size='tiny'),
        ]
        for line in lines:
            t0 =0.25+ line.write(begin_time=t0,transition_time=0.3)

        entries = [
            [
            SimpleTexBObject(r"\text{Source}"), SimpleTexBObject(r"\text{Projection}")
            ],
            [
            SimpleTexBObject(r"\vec{p}"),SimpleTexBObject(r"\vec{p}")
            ],
            [
                SimpleTexBObject(r"\vec{p}+\vec{e}_i"), SimpleTexBObject(r"\vec{q}")
            ],
            [
                SimpleTexBObject(r"\vec{p}+\vec{e}_i+\vec{e}_j"), SimpleTexBObject(r"\vec{r}")
            ],
            [
                SimpleTexBObject(r"\vec{p}+\vec{e}_j"), SimpleTexBObject(r"\vec{s}")
            ]
        ]
        table = Table(entries,location=[-4.5,0,2.25])

        t0 = 0.5+ table.write_all(begin_time=t0,transition_time=3)
        self.t0 = t0 + 1.5
    def face_selection(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 no_transmission_ray='True',
                                 rotation_euler=[0, 0, 260 * pi / 180])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine="BLENDER_EEVEE",
                               taa_render_samples=1024  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0.5)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        title = SimpleTexBObject(r"\text{Face Selection}",color='example',text_size='large',aligned='center',location=[0,0,3])
        t0 =0.5+ title.write(begin_time=t0,transition_time=1)


        # selected point
        point = Sphere(0.1,location=[0,0,0],name=r'\vec{p}',mesh='iso',
                       color='plastic_example',resolution=2,smooth=0,label_rotation=[pi/2,0,0])
        point.write_name_as_label(begin_time=t0+0.25,transition_time=0.25,modus='left',aligned='right')
        t0 = 0.5+point.grow(begin_time=t0,transition_time=0.5)


        origin = Vector([-4.5,0,-2.5])
        # unit vectors
        e0 = PArrow(start=origin,end=origin+Vector([0,-2,0]),name=r'\vec{e}_0')
        e0.grow(begin_time=t0,transition_time=1)
        e0_label = SimpleTexBObject(r"\vec{e}_0",location=origin+Vector([0.1,-2,0]))
        t0=e0_label.write(begin_time=t0+0.75,transition_time=1)-0.66666

        e1 = PArrow(start=origin, end=origin+Vector([2,0,0]), name=r'\vec{e}_1')
        e1.grow(begin_time=t0, transition_time=1)
        e1_label = SimpleTexBObject(r"\vec{e}_1", location=origin+Vector([2,0,-0.25]))
        t0 = e1_label.write(begin_time=t0 + 0.75, transition_time=1) - 0.3333

        e2 = PArrow(start=origin, end=origin+Vector([0,0,2]), name=r'\vec{e}_2')
        e2.grow(begin_time=t0, transition_time=1)
        e2_label = SimpleTexBObject(r"\vec{e}_2", location=origin+Vector([0.1,0,2]))
        t0 = e2_label.write(begin_time=t0 + 0.75, transition_time=1)

        title.rotate(rotation_euler=[-np.arctan2(-10,5),0,0],begin_time=t0,transition_time=3)
        t0=0.5+ibpy.camera_move(shift=[0,0,5],begin_time=t0,transition_time=3)

        point2 = Sphere(0.099,location=[0,0,0],name=r'\vec{p}+\vec{e}_0',mesh='iso',
                       color='plastic_example',resolution=2,smooth=0,label_rotation=[pi/2,0,0])
        point2.grow(begin_time=t0,transition_time=0)
        copy = e0.move_copy(direction=-origin, begin_time=t0, transition_time=1)
        t0 = point2.move(direction=[0,-2,0],begin_time=t0,transition_time=1)
        t0  = 0.5+ point2.write_name_as_label(begin_time=t0-0.25,transition_time=0.5,modus='left',aligned='right')

        point3 = Sphere(0.098, location=[0, -2, 0], name=r'\vec{p}+\vec{e}_0+\vec{e}_1', mesh='iso',
                        color='plastic_example', resolution=2, smooth=0, label_rotation=[pi / 2, 0, 0])
        point3.grow(begin_time=t0, transition_time=0)
        copy2 = e1.move_copy(direction=-origin + Vector([0, -2, 0]), begin_time=t0, transition_time=1)
        t0  = point3.move(direction=[2, 0, 0], begin_time=t0, transition_time=1)
        t0  = 0.5 + point3.write_name_as_label(begin_time=t0 - 0.25, transition_time=0.5, modus='right', aligned='left')

        point4 = Sphere(0.098, location=[0, 0, 0], name=r'\vec{p}+\vec{e}_1', mesh='iso',
                        color='plastic_example', resolution=2, smooth=0, label_rotation=[pi / 2, 0, 0])
        point4.grow(begin_time=t0, transition_time=0)
        copy3 = e1.move_copy(direction=-origin, begin_time=t0, transition_time=1)
        t0  = point4.move(direction=[2, 0, 0], begin_time=t0, transition_time=1)
        t0  = 0.5 + point4.write_name_as_label(begin_time=t0 - 0.25, transition_time=0.5, modus='right', aligned='left')

        face01 = Polygon([Vector(), Vector([0, -2, 0]), Vector([2, -2, 0]), Vector([2, 0, 0])],
                         color='plastic_x12_color', thickness=0.05)
        t0 = face01.grow(begin_time=t0, transition_time=1,pivot=Vector())
        copy.disappear(begin_time=t0-0.25,transition_time=0.5)
        copy2.disappear(begin_time=t0-0.25,transition_time=0.5)
        copy3.disappear(begin_time=t0-0.25,transition_time=0.5)
        point2.label_disappear(begin_time=t0-0.25,transition_time=0.5)
        point3.label_disappear(begin_time=t0-0.25,transition_time=0.5)
        t0 = 0.5+ point4.label_disappear(begin_time=t0-0.25,transition_time=0.5)

        title.rotate(rotation_euler=[-np.arctan2(-10, 5), 0,np.arctan2(5,10)], begin_time=t0, transition_time=3)
        t0 = 0.5 + ibpy.camera_move(shift=[5, 0, 0], begin_time=t0, transition_time=3)

        # face 02
        point5 = Sphere(0.099,location=[0,0,0],name=r'\vec{p}+\vec{e}_2',mesh='iso',
                       color='plastic_example',resolution=2,smooth=0,label_rotation=[pi/2,0,0])
        point5.grow(begin_time=t0,transition_time=0)
        copy4 = e2.move_copy(direction=-origin, begin_time=t0, transition_time=1)
        t0 = point5.move(direction=[0,0,2],begin_time=t0,transition_time=1)
        t0 = 0.5+ point5.write_name_as_label(begin_time=t0-0.25,transition_time=0.5,modus='left',aligned='right')

        point6 = Sphere(0.098, location=[0, 0, 2], name=r'\vec{p}+\vec{e}_0+\vec{e}_2', mesh='iso',
                        color='plastic_example', resolution=2, smooth=0, label_rotation=[pi / 2, 0, 0])
        point6.grow(begin_time=t0, transition_time=0)
        copy5 = e0.move_copy(direction=-origin+Vector([0,0,2]), begin_time=t0, transition_time=1)
        t0  =point6.move(direction=[0, -2, 0], begin_time=t0, transition_time=1)
        t0 = 0.5+ point6.write_name_as_label(begin_time=t0 - 0.25, transition_time=0.5, modus='left', aligned='right')

        face02 = Polygon([Vector(), Vector([0, 0, 2]), Vector([0, -2, 2]), Vector([0, -2, 0])],
                         color='plastic_x13_color', thickness=0.05)
        t0 = face02.grow(begin_time=t0, transition_time=1,pivot=Vector())
        copy4.disappear(begin_time=t0-0.25,transition_time=0.5)
        copy5.disappear(begin_time=t0-0.25,transition_time=0.5)
        point5.label_disappear(begin_time=t0 - 0.25, transition_time=0.5)
        t0 =0.5+point6.label_disappear(begin_time=t0 - 0.25, transition_time=0.5)

        # face 12

        point7 = Sphere(0.098, location=[0, 0, 2], name=r'\vec{p}+\vec{e}_1+\vec{e}_2', mesh='iso',
                        color='plastic_example', resolution=2, smooth=0, label_rotation=[pi / 2, 0, 0])
        point7.grow(begin_time=t0, transition_time=0)
        copy6 = e1.move_copy(direction=-origin + Vector([0, 0, 2]), begin_time=t0, transition_time=1)
        t0 = point7.move(direction=[2, 0, 0], begin_time=t0, transition_time=1)
        t0 = 0.5 + point7.write_name_as_label(begin_time=t0 - 0.25, transition_time=0.5, modus='right', aligned='left')

        face12 = Polygon([Vector(), Vector([2, 0, 0]), Vector([2, 0, 2]), Vector([0, 0, 2])],
                         color='plastic_x23_color', thickness=0.05)
        t0 = face12.grow(begin_time=t0, transition_time=1,pivot=Vector())
        copy6.disappear(begin_time=t0 - 0.25, transition_time=0.5)
        point7.label_disappear(begin_time=t0 - 0.25, transition_time=0.5)
        point.label_disappear(begin_time=t0 - 0.25, transition_time=0.5)

        # try projection

        u = Vector([0.408,-0.408,0.816])
        v = Vector([0.707,0.707,0])
        w = u.cross(v)

        face01.disappear(alpha=0.5,begin_time=t0,transition_time=0.5)
        face02.disappear(alpha=0.5,begin_time=t0,transition_time=0.5)
        t0 = 0.5+ face12.disappear(alpha=0.5,begin_time=t0,transition_time=0.5)

        arrowV = PArrow(start=Vector(), end=2 * v, name=r'\vec{v}', color='custom1')
        t0 = 0.5 + arrowV.grow(begin_time=t0, transition_time=1)
        arrowW = PArrow(start=Vector(), end=2 * w, name=r'\vec{n}', color='custom1')
        t0 = 0.5 + arrowW.grow(begin_time=t0, transition_time=1)
        u = w

        point2.move_to(target_location=Vector([0,-2,0])@u*u+Vector([0,-2,0])@v*v,begin_time=t0,transition_time=1)
        point3.move_to(target_location=Vector([2,-2,0])@u*u+Vector([2,-2,0])@v*v,begin_time=t0,transition_time=1)
        point4.move_to(target_location=Vector([2,0,0])@u*u+Vector([2,0,0])@v*v,begin_time=t0,transition_time=1)
        point5.move_to(target_location=Vector([0,0,2])@u*u+Vector([0,0,2])@v*v,begin_time=t0,transition_time=1)
        point6.move_to(target_location=Vector([0,-2,2])@u*u+Vector([0,-2,2])@v*v,begin_time=t0,transition_time=1)
        point7.move_to(target_location=Vector([2,0,2])@u*u+Vector([2,0,2])@v*v,begin_time=t0,transition_time=1)

        face01.morph_to(projector=lambda x:u.dot(x)*u+v.dot(x)*v,begin_time=t0,transition_time=1)
        face02.morph_to(projector=lambda x:u.dot(x)*u+v.dot(x)*v,begin_time=t0,transition_time=1)
        face01.disappear(alpha=1,begin_time=t0,transition_time=1)
        face02.disappear(alpha=1,begin_time=t0,transition_time=1)
        face12.disappear(alpha=1,begin_time=t0,transition_time=1)
        t0 = 0.5+ face12.morph_to(projector=lambda x:u.dot(x)*u+v.dot(x)*v,begin_time=t0,transition_time=1)


        self.t0 = t0
    def binaries(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 no_transmission_ray='True',
                                 rotation_euler=[0, 0, 260 * pi / 180])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine="BLENDER_EEVEE",
                               taa_render_samples=1024  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0.5)

        ref = ReferenceImage("ref2.png", scale=2.641, rotation_euler=[pi / 2, 0, 0])
        ref.appear(begin_time=t0, transition_time=0)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        for i in range(8):
            text = SimpleTexBObject(str(i)+r"\mapsto"+"("+str(i//4)+"|"+str(i%4//2)+"|"+str(i%2)+")",location = [-2+i//4*2,0,-0.4*(i%4)])
            text.write(begin_time=t0+0.5*i,transition_time=0.5)
        self.t0 = t0 + 1.5

    def projection_equations(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 no_transmission_ray='True',
                                 rotation_euler=[0, 0, 260 * pi / 180])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine="BLENDER_EEVEE",
                               taa_render_samples=1024  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0.5)

        ref = ReferenceImage("ref.png",scale=2.641,rotation_euler=[pi/2,0,0])
        ref.appear(begin_time=t0,transition_time=0)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        colors2 = flatten([['joker']*4,['text']*2 , ['example'] * 2, ['text']*5, ['drawing'] * 2,['text'],['joker']*4])
        eq =SimpleTexBObject(r"z_\text{min}<(\vec{p}-\vec{\sigma})\cdot \vec{n}\le z_\text{max}",location=[-2.2,0,2.8],color=colors2)
        eq.write(begin_time=t0,transition_time=1)

        self.t0 = t0 + 1.5

    def title(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 no_transmission_ray='True',
                                 rotation_euler=[0, 0, 260 * pi / 180])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine="BLENDER_EEVEE",
                               taa_render_samples=1024  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0.5)

        ref = ReferenceImage("ref.png", scale=2.641, rotation_euler=[pi / 2, 0, 0])
        ref.appear(begin_time=t0, transition_time=0)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        title = SimpleTexBObject(r"\text{The Projection of }\hspace{1ex} {\mathbb Z}^3 ", aligned='center',
                                 text_size='large', color='example', location=[0, 0, 3])
        title.write(begin_time=t0, transition_time=1)

        self.t0 = t0 + 1.5
    def projections(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 no_transmission_ray='True',
                                 rotation_euler=[0, 0, 260 * pi / 180])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine="BLENDER_EEVEE",
                               taa_render_samples=1024  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0.5)

        ref = ReferenceImage("ref.png",scale=2.641,rotation_euler=[pi/2,0,0])
        ref.appear(begin_time=t0,transition_time=0)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        colors1 = flatten([['text']*17,['custom1']*2,['text'],['example']*2,['text'],['custom1']*2,['text'],['example']*2,['text']])
        colors2 = flatten([['text']*17,['drawing']*2,['text'],['example']*2])


        lines = [
            SimpleTexBObject(r"\text{\frownie{}}", location=[5.4, 0, -2.55], text_size='Huge',color='important'),
            SimpleTexBObject(r"\text{\smiley{}}", location=[-5.4, 0, -2.55], text_size='Huge',color='joker',aligned='right'),
            SimpleTexBObject(r"\text{Plane projection:}\hspace{1ex} (\vec{u}\cdot \vec{p} | \vec{v}\cdot \vec{p} ) ",location=[-5.5,0,-3.5],color=colors1),
            SimpleTexBObject(r"\text{Ortho. projection:}\hspace{1ex} \vec{n}\cdot \vec{p}",location=[-2.2,0,2.8],color=colors2)
        ]

        for line in lines:
            t0 =0.5 + line.write(begin_time=t0,transition_time=0.5)



        self.t0 = t0 + 1.5

    def z3_annotations(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 no_transmission_ray='True',
                                 rotation_euler=[0, 0, 260 * pi / 180])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine="BLENDER_EEVEE",
                               taa_render_samples=1024  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0.5)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        locations = [
            [-5.6,0,-0.33],
            [-4.5,0,0.6],
            [-2.33,0,0.78],
        ]
        lines=[
            SimpleTexBObject(r"\text{range:}\hspace{1ex}r=3",location = locations[0],text_size='small'),
            SimpleTexBObject(r"\text{base:}\hspace{1ex}2\cdot r+1=7",location = locations[1],text_size='small'),
            SimpleTexBObject(r"7^3=343",location = locations[2],text_size='small'),
        ]

        for line in lines:
            t0 = 0.5+ line.write(begin_time=t0,transition_time=0.5)

        plane = Plane(u=[-0.35, 0.35], v=[-0.05, 0.05], color='example', rotation_euler=[pi / 2, 0, 0],
                      location=[2.5, 2.11 , 0])
        plane.grow(alpha=0.5, begin_time=t0 - 1, transition_time=0.5)

        self.t0 = t0+0.5

    def z3_ternary(self):
        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 no_transmission_ray='True',
                                 rotation_euler=[0, 0, 260 * pi / 180])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine="BLENDER_EEVEE",
                               taa_render_samples=1024  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0.5)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        examples = {0:"000_3",
                    1:"001_3",
                    2:"002_3",
                    3:"010_3",
                    6:"020_3",
                    9:"100_3",
                    10:"101_3",
                    26:"222_3"}

        count = 0
        shift = [0,0,0,0,0,0,-0.14,-0.14]
        for key,val in examples.items():
            text = SimpleTexBObject(str(key)+r"\mapsto"+val,location = [-0.5+shift[count],0,-0.75-0.4*count])
            text.write(begin_time=t0+0.5*count,transition_time=0.5)
            count+=1

        ypos = [3.34,3.21,3.085,2.95,2.56,2.18,2.06,0.02]
        for i in range(len(shift)):
            plane = Plane(u=[-1.4,1.4],v=[-0.05,0.05],color='example',rotation_euler=[pi/2,0,0],location = [0.075,ypos[i],0])
            plane.grow(alpha=0.5,begin_time=t0+0.5*i,transition_time=0.5)

        self.t0 = t0+5

    def z3_example(self):
        t0 = 0
        t0 = 0
        size = 5
        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 no_transmission_ray='True',
                                 rotation_euler=[0, 0, 260 * pi / 180])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine="BLENDER_EEVEE",
                               taa_render_samples=1024  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0.5)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)


        count = 0
        for x in range(-1,2):
            for y in range(-1,2):
                for z in range(-1,2):
                    text = SimpleTexBObject(r"("+str(x)+"|"+str(y)+"|"+str(z)+")",location = [-5+4*(count//14),0,3-0.5*(count%14)],aligned='center')
                    text.write(begin_time=t0+0.1*count,transition_time=0.5)
                    count+=1
        t0 = t0+0.1*27+1
        self.t0 = t0

    def z3_lattice(self):
        t0  =0
        t0 = 0
        size = 5
        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 no_transmission_ray='True',
                                 rotation_euler=[0, 0, 260 * pi / 180])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine="BLENDER_EEVEE",
                               taa_render_samples=1024  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0.5)

        ibpy.set_camera_location(Vector())
        empty = EmptyCube(location=[0, 0, 0])

        camera_circle = BezierCircle(location=[0, 0, 6.3], radius=np.sqrt(2) * 13)

        ibpy.set_camera_view_to(empty)
        ibpy.set_camera_follow(camera_circle)

        coords = CoordinateSystem(dim=3, lengths=[10, 10, 10], domains=[[-5, 5], [-5, 5], [-5, 5]],
                                  all_n_tics=[10, 10, 10],
                                  all_tic_labels=[np.arange(-5, 5.1, 1), np.arange(-5, 5.1, 1), np.arange(-5, 5.1, 1)],
                                  label_digits=[0, 0, 0],
                                  radii=[0.015, 0.015, 0.015],
                                  include_zeros=[False] * 3
                                  )
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=5)

        lattice_node = penrose_3D_analog(size=50, radius=0.05)
        lattice = Cube()
        # coords.add_object(lattice)

        lattice.add_mesh_modifier(type='NODES', node_group=lattice_node)

        x_max = get_geometry_node_from_modifier(lattice_node, label='xMax')
        x_min = get_geometry_node_from_modifier(lattice_node, label='xMin')
        y_max = get_geometry_node_from_modifier(lattice_node, label='yMax')
        y_min = get_geometry_node_from_modifier(lattice_node, label='yMin')
        z_min = get_geometry_node_from_modifier(lattice_node, label='zMin')
        z_max = get_geometry_node_from_modifier(lattice_node, label='zMax')

        max_material = ibpy.get_material_from_modifier(lattice_node, "MaxMaterial")
        min_material = ibpy.get_material_from_modifier(lattice_node, "MinMaterial")
        cube_material = ibpy.get_material_from_modifier(lattice_node, "CubeMaterial")
        icosphere_material = ibpy.get_material_from_modifier(lattice_node, "IcoSphereMaterial")
        materials = [max_material, min_material, cube_material]
        [ibpy.change_alpha_of_material(mat, from_value=0, to_value=0, begin_time=0, transition_time=0) for mat in
         materials]

        # trip to infinity
        r_slider = get_geometry_node_from_modifier(lattice_node, label='rMax')
        t0 = 0.5 + change_default_value(r_slider.outputs[0], from_value=0, to_value=50, begin_time=t0,
                                        transition_time=10)

        # restrict to box
        change_default_value(x_max.outputs[0], from_value=50, to_value=5, begin_time=t0, transition_time=5)
        change_default_value(x_min.outputs[0], from_value=-50, to_value=-5, begin_time=t0, transition_time=5)
        change_default_value(y_max.outputs[0], from_value=50, to_value=5, begin_time=t0, transition_time=5)
        change_default_value(y_min.outputs[0], from_value=-50, to_value=-5, begin_time=t0, transition_time=5)
        change_default_value(z_min.outputs[0], from_value=-50, to_value=-5, begin_time=t0, transition_time=5)
        t0 = 0.5 + change_default_value(z_max.outputs[0], from_value=50, to_value=5, begin_time=t0, transition_time=5)

        camera_circle.move_to(target_location=[0, 0, 1], begin_time=0, transition_time=t0 + 5)
        t0 = -5 + ibpy.camera_follow(camera_circle, initial_value=0, final_value=-0.5, begin_time=0,
                                     transition_time=t0 + 5)
        self.t0 = t0

    def z3(self):
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100, engine="BLENDER_EEVEE", taa_render_samples=64)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)
        t0 = 0.5

        title=SimpleTexBObject("{\mathbb Z}^3",color='plastic_example',text_size="Huge",thickness=10)
        title.write(begin_time=t0,transition_time=0.5)

        # create 10 random z3 points
        for i in range(10):
            coords = []
            for j in range(3):
                coords.append(random.randint(-5, 5))
            x = random.randint(-6,6)
            y = random.randint(-3,3)

            string = SimpleTexBObject(r"("+str(coords[0])+"|"+str(coords[1])+"|"+str(coords[2])+")",location=[x,0,y])
            t0 = string.write(begin_time=t0,transition_time=0.5)

        self.t0 = t0

    def ten_rotations(self):
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100, engine="BLENDER_EEVEE", taa_render_samples=64)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)
        t0 = 0.5


        space = "1"
        lines = [
            SimpleTexBObject(
                r"R_{01}\hspace{"+space+"em}  R_{02}\hspace{"+space+"em}  R_{03} \hspace{"+space+"em}  R_{04}\hspace{"+space+"em}  R_{12}\hspace{"+space+"em}  R_{13}\hspace{"+space+"em}  R_{14}\hspace{"+space+"em}  R_{23}\hspace{"+space+"em}  R_{24} \hspace{"+space+"em}  R_{34}",
                location=[-6, 0, 3], color='text',
                text_size='large', recreate=True)
        ]
        for line in lines:
            t0 = 0.1 + line.write(begin_time=t0, transition_time=1)

        self.t0 = t0

    def basis_rotation(self):
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100, engine="BLENDER_EEVEE", taa_render_samples=64)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)
        t0 = 0.5

        delta = 1.2
        lines = [
            SimpleTexBObject(r"R_{34}\cdot R_{24}\cdot R_{23} \cdot R_{14}\cdot R_{13}\cdot R_{12}\cdot R_{04}\cdot R_{03}\cdot R_{02} \cdot R_{01}\cdot \vec{u}", location=[-6, 0, 3], color='text',
                                 text_size='large', recreate=True),
            SimpleTexBObject(r"R_{34}\cdot R_{24}\cdot R_{23} \cdot R_{14}\cdot R_{13}\cdot R_{12}\cdot R_{04}\cdot R_{03}\cdot R_{02} \cdot R_{01}\cdot \vec{v}", location=[-6, 0, 3-delta], color='text',recreate=True,
                                 text_size='large'),
            SimpleTexBObject(
                r"R_{34}\cdot R_{24}\cdot R_{23} \cdot R_{14}\cdot R_{13}\cdot R_{12}\cdot R_{04}\cdot R_{03}\cdot R_{02} \cdot R_{01}\cdot \vec{w_1}", location=[-6, 0, 3 - 2*delta], color='text', recreate=True,
                text_size='large'),
            SimpleTexBObject(
                r"R_{34}\cdot R_{24}\cdot R_{23} \cdot R_{14}\cdot R_{13}\cdot R_{12}\cdot R_{04}\cdot R_{03}\cdot R_{02} \cdot R_{01}\cdot \vec{w_2}",
                location=[-6, 0, 3 - 3 * delta], color='text', recreate=True,
                text_size='large'),
            SimpleTexBObject(
                r"R_{34}\cdot R_{24}\cdot R_{23} \cdot R_{14}\cdot R_{13}\cdot R_{12}\cdot R_{04}\cdot R_{03}\cdot R_{02} \cdot R_{01}\cdot \vec{w_3}",
                location=[-6, 0, 3 - 4 * delta], color='text', recreate=True,
                text_size='large'),
            ]
        for line in lines:
            t0 = 0.1 + line.write(begin_time=t0, transition_time=1)

        self.t0 = t0
    def matrix_multiplication(self):
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100, engine="BLENDER_EEVEE", taa_render_samples=64)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)
        t0 = 0.5

        title = SimpleTexBObject(r"R_{01}\cdot \vec{v} = ", location=[-6, 0, 3], color='example',
                                 text_size='large')
        t0 = 0.5 + title.write(begin_time=t0, transition_time=1)

        rtimesv = SimpleTexBObject(
            r"\left(\begin{array}{c c c} \cos \vartheta &-\sin \vartheta & 0\\\sin \vartheta & \cos\vartheta & 0\\ 0 & 0 & 1\end{array}\right)\cdot \left(\begin{array}{c} v_x\\ v_y\\v_z\end{array}\right)=\left(\begin{array}{c} v_x\cos\vartheta -v_y\sin\vartheta\\ v_x\sin\vartheta+v_y\cos\vartheta \\ v_z\end{array}\right)",
            location=[-6, 0, 1.5], color='text')
        t0 = 0.25 + rtimesv.write(letter_set =[0,1,24,25,26,27,28,35,36,37,38,39,68,69],begin_time=t0, transition_time=0.5)
        t0 = 0.25 + rtimesv.write(letter_set =[2,4,6,9,11,13,15,18,20,21,29,32,30,33,31,34,40,42,44,46,49,50,53,56,58,60,62,64,66],begin_time=t0, transition_time=0.75)
        t0 = 0.25 + rtimesv.write(letter_set =[3,5,7,10,12,14,16,19,22,41,43,45,47,48,51,54,57,59,61,63,65,67],begin_time=t0, transition_time=0.5)
        t0 = 0.5 + rtimesv.write(letter_set =[8,17,23,52,55],begin_time=t0, transition_time=0.25)

        rtimesv2 = SimpleTexBObject(
            r"\left(\begin{array}{c} \vec{r_0}^\top\\ \vec{r_1}^\top\\ \vec{r_2}^\top\end{array}\right)\cdot \vec{v}=\left(\begin{array}{c} \vec{r_0}\cdot \vec{v}\\ \vec{r_1}\cdot \vec{v}\\\vec{r_2}\cdot \vec{v}\end{array}\right)",
            location=[-3.04,0,-0.5],color='text')
        t0 = 0.25+rtimesv2.write(letter_set=[0,1,14,15,16,17,18,19,20,21,40,41],begin_time=t0,transition_time=0.5)
        t0 = 0.25+rtimesv2.write(letter_set=[2,6,5,11,22,26,25,31,34,37],begin_time=t0,transition_time=0.5)
        t0 = 0.25+rtimesv2.write(letter_set=[3,8,7,12,23,28,27,32,35,38],begin_time=t0,transition_time=0.5)
        t0 = 0.5+rtimesv2.write(letter_set=[4,10,9,13,24,30,29,33,36,39],begin_time=t0,transition_time=0.5)
        self.t0 = t0

    def basis(self):
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100, engine="BLENDER_EEVEE", taa_render_samples=64)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)
        t0 = 0.5

        title = SimpleTexBObject(r"\text{Normalized basis vectors}",location=[-6,0,3],color='example',text_size='large')
        t0 = 0.5+ title.write(begin_time=t0,transition_time=1)

        u = SimpleTexBObject(r"{\vec{u}\over |\vec{u}|} = {1\over\sqrt{6}}\left(\begin{array}{c}-1\\-1\\2 \end{array}\right)\approx\left(\begin{array}{c} -0.41\\-0.41\\0.82\end{array}\right)",
                             location=[-5.75,0,1.5],color='custom1')
        t0 =0.1+ u.write(letter_set=[2,1,3,5,4,0,6,7],begin_time=t0,transition_time=0.3)
        t0 = 0.1+u.write(letter_set=[9,10,12,8,11,13,14,20,21,15,18,16,19,17,22],begin_time=t0,transition_time=0.5)
        t0 = 0.5+u.write(letter_set=[23,24,39,40,25,28,31,34,37,26,29,32,35,38,27,30,33,36],begin_time=t0,transition_time=0.5)

        v = SimpleTexBObject(
            r"{\vec{v}\over |\vec{v}|} = {1\over\sqrt{2}}\left(\begin{array}{c}-1\\1\\0 \end{array}\right)\approx\left(\begin{array}{c} -0.71\\0.71\\0\end{array}\right)",
            location=[-5.71, 0, -0.5], color='custom1',recreate=True)

        t0 = 0.1+v.write(letter_set=[2, 1, 3, 5, 4, 0, 6, 7], begin_time=t0, transition_time=0.3)
        t0 = 0.1+v.write(letter_set=[9, 10, 12, 8, 11, 13, 14, 19,20, 15, 18, 16, 17, 21], begin_time=t0,
                     transition_time=0.5)
        t0 = 0.5+v.write(letter_set=[22,23,  34,35, 24, 26, 28, 31, 33,25,27,29,32,30], begin_time=t0,
                     transition_time=0.5)


        n = SimpleTexBObject(r"\vec{n} = {1\over \sqrt{3}}\left(\begin{array}{c} 1\\1\\1\end{array}\right)\approx\left(\begin{array}{c} 0.58\\0.58\\0.58\end{array}\right)",
                             location = [-5.55,0,-2.32],color='drawing')

        t0 = 0.1 + n.write(letter_set=[ 1,0,2], begin_time=t0, transition_time=0.15)
        t0 = 0.1 + n.write(letter_set=[4,5,7,3,6,8,9,13,14,10,11,12,15], begin_time=t0,
                           transition_time=0.5)
        t0 = 0.5 + n.write(letter_set=[16,17,30,31,18,21,24,27,19,22,25,28,20,23,26,29], begin_time=t0,
                           transition_time=0.5)
        self.t0 = t0

    def rotations(self):
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100, engine="BLENDER_EEVEE", taa_render_samples=64)

        ibpy.set_camera_location(location=[0, -11, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)
        t0 = 0.5
        start = -6.5
        rot = SimpleTexBObject(
            r"R_{01}=\left(\begin{array}{ccccccc} \cos \vartheta_0 & -\sin \vartheta_0  & 0 \\ \sin \vartheta_0  & \cos \vartheta_0  & 0 \\ 0 & 0 & 1 \\\end{array}\right)",
             location=[start, 0, 2],color="x12_color")
        t0 = 0.5 + rot.write(
            letter_set=[0,1,2,3,4,5,6,8,10,13,15,17,19,21,22,26,28,29,7,9,11,14,16,18,20,23,25,27,30,12,24,31,32,33],
            begin_time=t0, transition_time=1)

        rot2 = SimpleTexBObject(
            r"R_{02}=\left(\begin{array}{ccccccc} \cos \vartheta_1 &0& -\sin \vartheta_1  \\ 0 & 1 & 0\\ \sin \vartheta_1  & 0 & \cos \vartheta_1 \\\end{array}\right)",
            location=[start+4.75, 0, 2], color="x13_color")
        t0 = 0.5 + rot2.write(
            letter_set=[0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 15, 17, 20, 22,24, 25, 29,31, 11, 18, 26, 7, 9, 12, 14, 16, 19, 21,23,27, 28, 30, 32,33],
            begin_time=t0, transition_time=1)

        rot3 = SimpleTexBObject(
            r"R_{12}=\left(\begin{array}{ccccccc} 1 & 0 & 0 \\ 0 & \cos \vartheta_2 & -\sin \vartheta_2  \\  0 &\sin \vartheta_2 & \cos \vartheta_2 \\\end{array}\right)",
            location=[start, 0, 0.5], color="x23_color")
        t0 = 0.5 + rot3.write(
            letter_set=[0, 1, 2, 3, 4, 5, 6, 13, 24, 7, 9, 11, 14, 16, 18, 20, 22, 25, 26,29,31, 8, 10, 12, 15, 17, 19,21, 23,27, 28,30,32, 33],
            begin_time=t0, transition_time=1)

        t0 = rot3.disappear(begin_time=t0,transition_time=0.5)
        t0 = 0.5+ rot2.disappear(begin_time=t0,transition_time=0.5)

        rot5 =  SimpleTexBObject(
            r"R_{01}=\left(\begin{array}{ccccccc} \cos \vartheta_0 & -\sin \vartheta_0  & 0  & 0 & 0 \\ \sin \vartheta_0  & \cos \vartheta_0  & 0 & 0 & 0\\ 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 &0 \\ 0 & 0 & 0 & 0 & 1\end{array}\right)",
             location=[start, 0, 2],color="x12_color")

        rot.letters[4].move(direction=[0,0.3809,0],begin_time=t0,transition_time=0.2)
        rot.letters[5].move(direction=[0,-0.3809,0],begin_time=t0,transition_time=0.2)
        rot.letters[32].move(direction=[0.9654, 0.3809, 0], begin_time=t0, transition_time=0.2)
        t0 = rot.letters[33].move(direction=[0.9654, -0.3809, 0], begin_time=t0, transition_time=0.2)
        rot5.write(letter_set=[4,5,6,7],begin_time=t0,transition_time=0.2)
        t0  =rot5.write(letter_set=[54,55,56,57],begin_time=t0,transition_time=0.2)

        [rot.letters[l].move(direction=[0,0.381,0],begin_time=t0+0.05*l,transition_time=0.2) for l in [6,8,10,13,15,17,19,21,22,26,28,29,7,9,11,14,16,18,20,23,25,27,30,12,24,31]]
        t0 +=0.2+0.05*26
        t0  =0.5+rot5.write(letter_set=[42,47,43,48,44,49,17,30,40,45,50,18,31,41,46,51],begin_time=t0,transition_time=1)


        self.t0 = t0

    def the_wall(self):

        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100, engine="BLENDER_EEVEE", taa_render_samples=64)

        ibpy.set_camera_location(location=[0, -10, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        # add plane, whose mesh vertices will be the centers of the tilings
        plane = Plane(resolution=[10, 10], uniformization=False,rotation_euler=[pi/2,0,0])
        z5_nodes = de_bruijn(k=30, base_color='joker', tile_separation=0.05, radius=76, emission=0.15,scaling=[152,153,0])
        plane.add_mesh_modifier(type='NODES', node_group=z5_nodes)
        set_material(plane, "Penrose", slot=0)
        plane.add_mesh_modifier('SOLIDIFY', thickness=0.12, offset=0)
        #plane.add_mesh_modifier('BEVEL', amount=0.05, segments=2)

        plane.appear(begin_time=t0, transition_time=0)

        tile_size = get_geometry_node_from_modifier(z5_nodes, 'tile_size')
        change_default_value(tile_size.inputs['Scale'], from_value=0, to_value=0.95, begin_time=0, transition_time=2)
        stretcher = get_geometry_node_from_modifier(z5_nodes, 'stretcher')
        t0 = change_default_value(stretcher.inputs['Scale'], from_value=10, to_value=1, begin_time=0,
                                        transition_time=5)
        t0 =  ibpy.camera_move(shift=[0,-220,0],begin_time=t0,transition_time=5)
        ibpy.camera_move(shift=[0,130,0],begin_time=t0,transition_time=5)
        ibpy.make_image_material(src='logo.png', name='PenroseImage')

        mixer = ibpy.get_nodes_of_material(plane, name_part='Mix')[0]

        t0 = 0.5+ change_default_value(mixer.inputs[0], from_value=1, to_value=0, begin_time=t0, transition_time=5)


        self.t0 = t0

    def the_end_of_the_wall(self):

        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100, engine="BLENDER_EEVEE", taa_render_samples=64)

        ibpy.set_camera_location(location=[0, -100, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        plane = Plane(resolution=[10, 10], uniformization=False,rotation_euler=[pi/2,0,0])
        #z5_nodes = de_bruijn(k=15, base_color='joker', tile_separation=0.05, radius=51, emission=0.15,scaling=[152,153,0])
        z5_nodes = de_bruijn(k=30, base_color='joker', tile_separation=0.05, radius=76, emission=0.15,scaling=[152,153,0])
        plane.add_mesh_modifier(type='NODES', node_group=z5_nodes)
        set_material(plane, "Penrose", slot=0)
        plane.add_mesh_modifier('SOLIDIFY', thickness=0.12, offset=0)

        plane.appear(begin_time=t0, transition_time=0)

        tile_size = get_geometry_node_from_modifier(z5_nodes, 'tile_size')
        change_default_value(tile_size.inputs['Scale'], from_value=0.95, to_value=0.95, begin_time=0, transition_time=0)
        stretcher = get_geometry_node_from_modifier(z5_nodes, 'stretcher')
        change_default_value(stretcher.inputs['Scale'], from_value=1, to_value=1, begin_time=0, transition_time=0)
        ibpy.make_image_material(src='logo.png', name='PenroseImage')
        mixer = ibpy.get_nodes_of_material(plane, name_part='Mix')[0]
        t0 = change_default_value(mixer.inputs[0], from_value=0, to_value=0, begin_time=0, transition_time=0)

        ibpy.apply_modifiers(plane)
        ibpy.separate(plane, type='LOOSE')
        make_rigid_body(plane, all_similar_objects=True, use_margin=True,
                        collision_margin=0.001, friction=0.25)

        self.t0 = t0

if __name__ == '__main__':
    try:
        example = Penrose()
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
