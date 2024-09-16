import os
from copy import deepcopy
from datetime import date, datetime

import bpy
import bmesh

import mathutils
import numpy as np
from mathutils import Vector, Quaternion, Matrix

from compositions.compositions import create_composition
from interface.interface_constants import EMISSION, TRANSMISSION, BLENDER_EEVEE, blender_version

from utils.constants import BLEND_DIR, FRAME_RATE, OBJECT_APPEARANCE_TIME, OSL_DIR, COLOR_NAMES, COLORS_SCALED, IMG_DIR, \
    DEFAULT_ANIMATION_TIME, RES_HDRI_DIR, FINAL_DIR, VID_DIR, COLOR_PREFIXES, SPECIALS, COLORS, APPEND_DIR
from utils.geometry import BoundingBox
from utils.kwargs import get_from_kwargs
from utils.mathematics import lin_map
from utils.string_utils import remove_digits

"""
This interface encodes most of the blender functionality.
Therefore changes in the api only have to be implemented in this interface.
It is also a convenient reference to look up commands as they should be used in this package.

TODO may wishful thinking
If it was possible to keep track of all objects in such an interface,
one could generate an actual blender script that reproduces the functionality of this package.

"""
EPS = 0.0001
CAMERA_FOLLOW_PATHS = []
CAMERA_FOLLOW_PATH_TARGET_DICTIONARY = {}

FOLLOW_PATH_DICTIONARY = {}
TRACK_TO_DICTIONARY = {}

SOCKET_TYPES = (
'FLOAT', 'INT', 'BOOLEAN', 'VECTOR', 'ROTATION', 'STRING', 'RGBA', 'OBJECT', 'IMAGE', 'GEOMETRY', 'COLLECTION',
'TEXTURE', 'MATERIAL')
DATA_TYPES = ('FLOAT', 'INT', 'FLOAT_VECTOR', 'FLOAT_COLOR', 'BYTE_COLOR', 'BOOLEAN', 'FLOAT2', 'QUATERNION')

# where as '*' is the ordinary multiplication for scalars, 'mul' is the corresponding vector operator
OPERATORS = ['*', 'mul', '%', 'mod', '/', 'div', '+', 'add', '-', 'sub', '**', 'sin', 'cos', 'tan', '^', 'lg',
             'sqrt', 'exp', 'abs', 'min', 'max', '<', '>', 'sgn', 'round', 'floor', 'vfloor', 'ceil',
             'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'length', 'scale', 'sqrt', '=', 'dot',
             'cross', 'rot', 'axis_rot', 'rot2euler', 'axis_angle_euler', 'not', 'normalize', 'and', 'or']
# operators that return data of type VECTOR
VECTOR_OPERATORS = ['mul', 'mod', 'div', 'add', 'sub', 'scale', 'vfloor', 'cross', 'rot', 'axis_rot', 'rot2euler',
                    'axis_angle_euler', 'normalize']


def get_context():
    return bpy.context


def get_scene():
    return get_context().scene


def close(x, y, precision=EPS):
    return np.abs(x - y) / precision < 1


def get_obj(bob):
    if hasattr(bob, 'ref_obj'):
        obj = bob.ref_obj
        return get_obj(obj)  # recursive call, since there can be cascading BObjects
    else:
        return bob


def get_obj_from_name(name=None):
    """finds the object of a given name among all objects"""
    if name:
        for o in bpy.data.objects:
            if name == o.name:
                return o


def rename(bob, name):
    obj = get_obj(bob)
    obj.name = name


###########
# objects #
###########

def create(mesh, name, location):
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.ops.object.shade_smooth()
    return obj


##########
# light #
##########

def set_shadow(value):
    for light in bpy.data.lights:
        light.cycles.cast_shadow = value


def set_shadow_of_object(b_object, shadow=True):
    """
    :param b_object:
    :param shadow:
    :return:
    """
    obj = get_obj(b_object)
    obj.visible_shadow = shadow
    obj.visible_glossy = shadow


def set_sun_light(location=[0, 0, 10], energy=2):
    light = bpy.data.objects['Sun']
    light.location = location

    light.data.energy = energy


def remove_sun_light():
    if 'Sun' in bpy.data.objects:
        sun = bpy.data.objects['Sun']
        bpy.data.objects.remove(sun)


def add_light_probe(**kwargs):
    if 'type' in kwargs:
        probe_type = kwargs.pop('type')
    else:
        probe_type = 'PLANAR'
    bpy.ops.object.lightprobe_add(type=probe_type, **kwargs)
    probe = bpy.context.object
    return probe


def add_spot_light(**kwargs):
    '''
    :param kwargs: location,radius,scale,energy
    :return:
    '''
    if 'energy' in kwargs:
        energy = kwargs['energy']
    else:
        energy = 10
    kwargs.pop('energy')

    bpy.ops.object.light_add(type='SPOT', **kwargs)
    spot = bpy.context.object
    spot.data.energy = energy
    return spot


def add_area_light(**kwargs):
    '''
    :param kwargs: location,radius,scale,energy
    :return:
    '''

    color = get_from_kwargs(kwargs, 'color', 'text')
    energy = get_from_kwargs(kwargs, 'energy', 10)
    shape = get_from_kwargs(kwargs, 'shape', 'SQUARE')
    size = get_from_kwargs(kwargs, 'size', 1)
    size_y = get_from_kwargs(kwargs, 'size_y', 1)
    diffuse_factor = get_from_kwargs(kwargs, 'diffuse_factor', 1)
    specular_factor = get_from_kwargs(kwargs, 'specular_factor', 1)
    volume_factor = get_from_kwargs(kwargs, 'volume_factor', 1)

    bpy.ops.object.light_add(type='AREA', **kwargs)
    area = bpy.context.object
    area.data.energy = energy
    area.data.color = get_color_from_string(color)[0:3]
    area.data.shape = shape
    area.data.size = size
    area.data.size_y = size_y
    area.data.diffuse_factor = diffuse_factor
    area.data.specular_factor = specular_factor
    area.data.volume_factor = volume_factor

    return area


def add_point_light(**kwargs):
    '''
    :param kwargs: location,radius,scale,energy
    :return:
    '''
    if 'energy' in kwargs:
        energy = kwargs['energy']
    else:
        energy = 10
    kwargs.pop('energy')

    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1] * 4
    kwargs.pop('color')

    bpy.ops.object.light_add(type='POINT', **kwargs)
    point = bpy.context.object
    point.data.energy = energy
    point.data.color = color
    return point


def switch_on(bob, begin_frame=0, frame_duration=1):
    obj = bob.ref_obj
    obj.data.energy = 0
    insert_keyframe(obj.data, 'energy', begin_frame)
    obj.data.energy = bob.energy
    insert_keyframe(obj.data, 'energy', begin_frame + np.maximum(1, frame_duration))


def switch_off(bob, begin_frame=0, frame_duration=1):
    obj = bob.ref_obj
    obj.data.energy = get_energy_at_frame(bob, begin_frame)
    insert_keyframe(obj.data, 'energy', begin_frame)
    obj.data.energy = 0
    insert_keyframe(obj.data, 'energy', begin_frame + np.maximum(1, frame_duration))


def get_energy_at_frame(bob, frame):
    set_frame(frame)
    obj = bob.ref_obj
    return obj.data.energy


def get_parameter_at_frame(function, frm):
    function = get_obj(function)
    set_frame(frm)
    end = function.data.bevel_factor_end
    return end


def get_value_at_frame(function, frm):
    set_frame(frm)
    return function.value


##########
# camera #
##########


def add_camera(location=[0, 0, 0], rotation=[0, 0, 0]):
    bpy.ops.object.camera_add(location=location, rotation=rotation)

    # the order of the following two constraints is important
    for i in range(10):
        bpy.ops.object.constraint_add(type='FOLLOW_PATH')  # add the ability to follow a path
        constraint = bpy.data.objects['Camera'].constraints[-1]
        if i > 0:
            constraint.influence = 0  # set all but the first path to zero influence
            insert_keyframe(constraint, 'influence', 0)
        CAMERA_FOLLOW_PATHS.append(constraint)  # add the newly created follow path to the list for later access
    bpy.ops.object.constraint_add(type='TRACK_TO')  # add the ability to track the view

    # cam.type = 'ORTHO'
    scn = get_scene()
    scn.camera = bpy.context.object
    cam = bpy.data.cameras[0]
    cam.type = 'PERSP'
    cam.ortho_scale = 30
    cam.lens = 30


def set_camera_location(location=[0, -20, 0], frame=0):
    cam = get_camera()
    cam.location = location
    cam.keyframe_insert(data_path='location', frame=frame)


def set_camera_rotation(rotation=[0, 0, 0]):
    cam = get_camera()
    cam.rotation_euler = rotation


def set_camera_lens(lens=50, clip_end=1000):
    cam = get_camera()
    cam.data.lens = lens
    cam.data.clip_end = clip_end


def camera_zoom(lens=50, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
    cam = get_camera()
    frame = int(begin_time * FRAME_RATE)
    set_frame(frame)
    lens_old = cam.data.lens
    insert_keyframe(cam.data, 'lens', frame)
    cam.data.lens = lens
    insert_keyframe(cam.data, 'lens', begin_time * FRAME_RATE + np.maximum(1, transition_time * FRAME_RATE))
    return begin_time + transition_time


def camera_move(shift, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME, verbose=False):
    cam = get_camera()
    start_frame = begin_time * FRAME_RATE
    location = get_location_at_frame(cam, start_frame)
    cam.location = location
    cam.keyframe_insert(data_path='location', frame=start_frame)
    for i, s in enumerate(shift):
        cam.location[i] += s
    cam.keyframe_insert(data_path='location', frame=(begin_time + transition_time) * FRAME_RATE)
    if verbose:
        print("Camera move by: " + str(shift) + " at " + str(begin_time))
    return begin_time + transition_time


def camera_rotate_to(rotation_euler, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
    cam = get_camera()
    cam.keyframe_insert(data_path='rotation_euler', frame=begin_time * FRAME_RATE)
    for i, s in enumerate(rotation_euler):
        cam.rotation_euler[i] = s
    cam.keyframe_insert(data_path='rotation_euler', frame=(begin_time + transition_time) * FRAME_RATE)


def get_camera():
    return bpy.data.objects['Camera']


def set_camera_copy_location(target=None, **kwargs):
    '''
    follow an object without rotation of the camera
    :param target:
    :param kwargs:
    :return:
    '''
    if 'offset' in kwargs:
        offset = kwargs.pop('offset')
    else:
        offset = False
    cam = get_camera()

    if target is not None:
        constraint = set_copy_location(cam, target)
        constraint.use_offset = offset


def set_camera_view_to(target=None, rotation_euler=[0, 0, 0], targetZ=False, up_axis='UP_Y'):
    '''
    :param influence:
    :param target:
    :param rotation_euler:
    :param targetZ: if true the up direction is given by the target z-direction instead of the world z-direction
    :return:
    '''
    cam = get_camera()
    cam.rotation_euler = rotation_euler
    if target is not None:
        constraint = set_track(cam, target)
        constraint.use_target_z = targetZ
        constraint.up_axis = up_axis


def set_camera_follow(target=None):
    constraint = CAMERA_FOLLOW_PATHS[len(CAMERA_FOLLOW_PATH_TARGET_DICTIONARY)]
    CAMERA_FOLLOW_PATH_TARGET_DICTIONARY[target] = constraint

    if target:
        constraint.target = target.ref_obj
        constraint.use_fixed_location = True
        constraint.use_curve_radius = True
        constraint.use_curve_follow = True


def set_follow(bob, target=None, **kwargs):
    obj = get_obj(bob)
    set_active(obj)
    bpy.ops.object.constraint_add(type='FOLLOW_PATH')
    use_fixed_location = get_from_kwargs(kwargs, 'use_fixed_location', True)
    use_curve_radius = get_from_kwargs(kwargs, 'use_curve_radius', True)
    use_curve_follow = get_from_kwargs(kwargs, 'use_curve_follow', True)
    constraint = obj.constraints[-1]
    FOLLOW_PATH_DICTIONARY[(bob, target)] = constraint
    if target:
        constraint.target = target.ref_obj
        constraint.use_fixed_location = use_fixed_location
        constraint.use_curve_radius = use_curve_radius
        constraint.use_curve_follow = use_curve_follow


def camera_change_track_influence(target, start, end, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
    camera = get_camera()
    c = TRACK_TO_DICTIONARY[(camera, target)]
    c.influence = start
    insert_keyframe(c, 'influence', begin_time * FRAME_RATE)
    c.influence = end
    insert_keyframe(c, 'influence', (begin_time + transition_time) * FRAME_RATE)
    return begin_time + transition_time


def camera_change_follow_influence(target, start, end, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
    camera = get_camera()
    c = CAMERA_FOLLOW_PATH_TARGET_DICTIONARY[target]
    c.influence = start
    insert_keyframe(c, 'influence', begin_time * FRAME_RATE)
    c.influence = end
    insert_keyframe(c, 'influence', (begin_time + transition_time) * FRAME_RATE)


def set_track_influence(bob, target, influence, begin_time=0):
    obj = get_obj(bob)
    c = TRACK_TO_DICTIONARY[(obj, target)]
    c.influence = influence
    insert_keyframe(c, 'influence', begin_time * FRAME_RATE)


def set_follow_influence(bob, target, value, begin_time=0):
    c = FOLLOW_PATH_DICTIONARY[(bob, target)]
    c.influence = value
    insert_keyframe(c, 'influence', begin_time * FRAME_RATE)


def change_follow_influence(bob, target, initial, final, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
    c = FOLLOW_PATH_DICTIONARY[(bob, target)]
    c.influence = initial
    insert_keyframe(c, 'influence', begin_time * FRAME_RATE)
    c.influence = final
    insert_keyframe(c, 'influence', begin_time * FRAME_RATE + np.maximum(1, int(transition_time * FRAME_RATE)))


def set_copy_location(bob, target=None):
    obj = get_obj(bob)
    set_active(obj)
    bpy.ops.object.constraint_add(type='COPY_LOCATION')
    constraint = obj.constraints[-1]
    if target:
        constraint.target = target.ref_obj
    return constraint


def set_track(bob, target=None):
    obj = get_obj(bob)
    set_active(obj)
    bpy.ops.object.constraint_add(type='TRACK_TO')
    constraint = obj.constraints[-1]
    TRACK_TO_DICTIONARY[(obj, target)] = constraint
    if target:
        constraint.target = target.ref_obj
    return constraint


def follow(bob, target, initial_value, final_value, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME, **kwargs):
    constraint = FOLLOW_PATH_DICTIONARY[(bob, target)]
    begin_frame = begin_time * FRAME_RATE
    constraint.offset_factor = initial_value
    if 'forward_axis' in kwargs:
        constraint.forward_axis = kwargs['forward_axis']
    if 'up_axis' in kwargs:
        constraint.up_axis = kwargs['up_axis']
    insert_keyframe(constraint, 'offset_factor', begin_frame)
    constraint.offset_factor = final_value
    insert_keyframe(constraint, 'offset_factor', (begin_time + transition_time) * FRAME_RATE)
    return begin_time + transition_time


def camera_follow(target, initial_value, final_value, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
    constraint = CAMERA_FOLLOW_PATH_TARGET_DICTIONARY[target]
    for c in CAMERA_FOLLOW_PATHS:
        begin_frame = begin_time * FRAME_RATE
        set_frame(
            begin_frame)  # this line is important to jump from the current value of the constraint's influence to the new value
        if c != constraint:
            insert_keyframe(c, 'influence', begin_frame)
            c.influence = 0
            insert_keyframe(c, 'influence', begin_frame + 1)
        else:
            insert_keyframe(c, 'influence', begin_frame)
            c.influence = 1
            insert_keyframe(c, 'influence', begin_frame + 1)

            c.offset_factor = initial_value
            insert_keyframe(c, 'offset_factor', begin_frame + 1)
            c.offset_factor = final_value
            insert_keyframe(c, 'offset_factor', (begin_time + transition_time) * FRAME_RATE)
    return begin_time + transition_time


###############
# meshes
###############

def smooth_mesh(bob):
    obj = get_obj(bob)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for f in bm.faces:
        f.smooth = True
    bm.to_mesh(obj.data)
    bm.free()


def analyse_mesh(bob):
    ob = get_obj(bob)
    context = bpy.context
    dg = context.evaluated_depsgraph_get()

    bm = bmesh.new()
    bm.from_object(ob, dg)
    print("Evaluated bmesh object", ob.name, "mesh:", ob.data.name)
    tris = bm.calc_loop_triangles()
    print(
        "faces", len(bm.faces),
        "edges", len(bm.edges),
        "verts", len(bm.verts),
        "tris", len(tris)
    )

    for name, cl in bm.loops.layers.color.items():
        print("Colour Layer", name)

        for i, tri in enumerate(tris):
            print("Tri", i)
            for loop in tri:
                print(
                    "loop", loop.index,
                    "face", loop.face.index,
                    "edge", loop.edge.index,
                    "vert", loop.vert.index,
                    "colour", loop[cl][:]
                )


def apply(bob, location=True, rotation=True, scale=True):
    """
    map local coordinates into shape key coordinates
    :param scale:
    :param rotation:
    :param location:
    :param bob:
    :return:
    """
    o = get_obj(bob)
    set_active(o)
    select(o, deselect_others=True)
    bpy.ops.object.transform_apply(location=location, rotation=rotation, scale=scale)


def convert_to_mesh(obj, apply_transform=False):
    obj = get_obj(obj)
    if isinstance(obj, list):
        first = True
        for bob in obj:
            o = get_obj(bob)
            if first:
                set_active(o)  # make the first object to the active object
                first = False
            select(o, deselect_others=False)
            bpy.ops.object.convert(target='MESH')
            if apply_transform:
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    else:
        obj = get_obj(obj)
        select(obj)
        bpy.ops.object.convert(target='MESH')


def add_empty(**kwargs):
    bpy.ops.object.empty_add(**kwargs)
    return bpy.context.object


def create_vertex_group(bob, vertex_indices, name):
    group = bob.ref_obj.vertex_groups.new(name=name)
    group.add(vertex_indices, 1.0, 'ADD')
    return group


def shade_smooth(auto=True):
    # bpy.ops.object.shade_smooth(use_auto_smooth=auto)
    # adjustments for blender 4.1
    bpy.ops.object.shade_smooth()


def add_sphere(radius=1, location=(0, 0, 0), scale=(1, 1, 1), resolution=5, smooth=True, **kwargs):
    deselect_all()

    if 'mesh_type' in kwargs:
        mesh_type = kwargs['mesh_type']
    else:
        mesh_type = 'uv'

    if mesh_type == 'uv':
        segments = pow(2, resolution)
        rings = int(segments / 2)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=segments, ring_count=rings, location=location,
                                             scale=scale, align='WORLD')
    else:
        bpy.ops.mesh.primitive_ico_sphere_add(radius=radius, location=location, subdivisions=resolution, scale=scale,
                                              align='WORLD')
    if smooth:
        shade_smooth()
    return bpy.context.object


def add_cylinder(smooth=True, **kwargs):
    deselect_all()
    bpy.ops.mesh.primitive_cylinder_add(**kwargs)
    if smooth:
        shade_smooth()
    cyl = bpy.context.object
    return cyl


def add_torus(smooth=True, **kwargs):
    deselect_all()
    bpy.ops.mesh.primitive_torus_add(**kwargs)
    if smooth:
        shade_smooth()
    tor = bpy.context.object
    return tor


def add_reference_image(name, **kwargs):
    path = os.path.join(IMG_DIR, name)
    if blender_version() < (4, 2):
        bpy.ops.object.load_reference_image(filepath=path)
    else:
        bpy.ops.object.empty_image_add(filepath=path)
    return bpy.context.active_object


def add_cube(smooth=True, **kwargs):
    deselect_all()
    bpy.ops.mesh.primitive_cube_add(**kwargs)
    if smooth:
        shade_smooth()
    return bpy.context.object


def add_cone(smooth=True, **kwargs):
    deselect_all()
    bpy.ops.mesh.primitive_cone_add(**kwargs)
    if smooth:
        shade_smooth()
    return bpy.context.object


def add_circle(**kwargs):
    deselect_all()
    if 'extrude' in kwargs:
        extrude = kwargs['extrude']
        kwargs.pop('extrude')
    else:
        extrude = 0
    if 'bevel' in kwargs:
        bevel = kwargs['bevel']
        kwargs.pop('bevel')
    else:
        bevel = 0
    if 'resolution' in kwargs:
        resolution = kwargs.pop('resolution')
    else:
        resolution = 12
    bpy.ops.curve.primitive_bezier_circle_add(**kwargs)
    bpy.context.object.data.bevel_depth = bevel
    bpy.context.object.data.extrude = extrude
    bpy.context.object.data.splines[0].resolution_u = int(resolution)
    return bpy.context.object


def add_plane(name="Plane", smooth=True):
    deselect_all()
    bpy.ops.mesh.primitive_plane_add()  # Adds in a plane to work with
    if smooth:
        bpy.ops.object.shade_smooth()
    plane = bpy.context.object
    plane.name = name
    return plane


def create_mesh(vertices, edges=[], faces=[], name='mesh'):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, edges, faces)
    mesh.update()
    return mesh


def add_wind(**kwargs):
    deselect_all()
    bpy.ops.object.effector_add(type='WIND', enter_editmode=False, align='WORLD', **kwargs)
    return bpy.context.object


def add_turbulence(**kwargs):
    deselect_all()
    bpy.ops.object.effector_add(type='TURBULENCE', enter_editmode=False, align='WORLD', **kwargs)
    return bpy.context.object


def add_force(**kwargs):
    deselect_all()
    bpy.ops.object.effector_add(type='FORCE', enter_editmode=False, align='WORLD', **kwargs)
    return bpy.context.object


def decay_force(bob, initial_strength, final_strength=0, begin_frame=0, frame_duration=60):
    obj = get_obj(bob)
    obj.field.strength = initial_strength
    insert_keyframe(obj.field, 'strength', begin_frame)
    obj.field.strength = final_strength
    insert_keyframe(obj.field, 'strength', begin_frame + frame_duration)


# set edit type
def set_edit_mode():
    bpy.ops.object.mode_set(mode='EDIT')


def set_object_mode():
    bpy.ops.object.mode_set(mode='OBJECT')


# set active object and get active object

def set_active(obj):
    obj = get_obj(obj)
    bpy.context.view_layer.objects.active = obj


def get_active():
    return bpy.context.view_layer.objects.active
    # alternatively return bpy.context.active_object


# selections and de-selections

def select(obj, deselect_others=True):
    if deselect_others:
        deselect_all()
    set_select(obj)


def select_recursively(obj):
    deselect_all()
    deep_select(obj)


def select_all_deep(bobs):
    deselect_all()
    for bob in bobs:
        deep_select(bob.ref_obj)


def deep_select(obj):
    try:
        set_select(obj)
    except:
        print(obj.name, " not found for copying")
    for child in obj.children:
        deep_select(child)


def un_select(obj):
    obj.select_set(False)


def deselect_all():
    bpy.ops.object.select_all(action='DESELECT')


def set_select(obj, value=True):
    obj.select_set(value)


def set_several_select(objects, value=True):
    bpy.ops.object.select_all(action='DESELECT')
    for bob in objects:
        o = get_obj(bob)
        o.select_set(True)


# hide and unhide

def is_hidden(obj):
    return obj.hide_get()


def un_hide(obj):
    obj.hide_set(False)


def hide(bob, begin_time=0):
    hide_rec(bob.ref_obj, begin_time=begin_time)


def hide_rec(obj, begin_time=0):
    """
    hide b_object from render view
    it works for mesh and curve objects

    :param bobj:
    :param begin_time:
    :return:
    """
    obj = get_obj(obj)
    frame = int(begin_time * FRAME_RATE)
    if hasattr(obj, 'hide_render'):
        obj.hide_render = False
        insert_keyframe(obj, 'hide_render', frame - 1)
        obj.hide_render = True
        insert_keyframe(obj, 'hide_render', frame)
    elif hasattr(obj, 'data'):
        if hasattr(obj.data, 'hide_rander'):
            obj.data.hide_render = False
            insert_keyframe(obj, 'hide_render', frame - 1)
            obj.data.hide_render = True
            insert_keyframe(obj, 'hide_render', frame)
    # obj.hide_set(True)
    for child in obj.children:
        hide_rec(child, begin_time=begin_time)


def unhide_frm(bob, frame):
    unhide_frm_rec(bob.ref_obj, frame)


def hide_frm(bob, frame):
    hide_frm_rec(bob.ref_obj, frame)


def unhide(bob, begin_time=0):
    unhide_rec(bob.ref_obj, begin_time=begin_time)


def set_hide(bob, hide, frame):
    obj = get_obj(bob)
    set_frame(frame)

    obj.hide_render = hide
    insert_keyframe(obj, 'hide_render', frame)
    obj.hide_viewport = hide
    insert_keyframe(obj, 'hide_viewport', frame)


def unhide_rec(obj, begin_time=0):
    """
    hide b_object from render view until the time begin_time
    it works for mesh and curve objects

    :param bobj:
    :param begin_time:
    :return:
    """
    obj = get_obj(obj)
    frame = int(begin_time * FRAME_RATE)
    if hasattr(obj, 'hide_render'):
        obj.hide_render = True
        insert_keyframe(obj, 'hide_render', frame - 1)
        obj.hide_render = False
        insert_keyframe(obj, 'hide_render', frame)
    elif hasattr(obj, 'data'):
        if hasattr(obj.data, 'hide_rander'):
            obj.data.hide_render = True
            insert_keyframe(obj, 'hide_render', frame - 1)
            obj.data.hide_render = False
            insert_keyframe(obj, 'hide_render', frame)

    for child in obj.children:
        unhide_rec(child, begin_time=begin_time)
    # obj.hide_set(True)


def unhide_frm_rec(obj, frame):
    """
    hide b_object from render view until the time begin_time
    it works for mesh and curve objects

    :param obj:
    :param frame:
    :return:
    """
    obj = get_obj(obj)
    if hasattr(obj, 'hide_render'):
        # don't uncomment this, it destroys the possiblity to merge strings
        # obj.hide_viewport = True
        # insert_keyframe(obj, 'hide_viewport', frame-1)
        obj.hide_render = True
        insert_keyframe(obj, 'hide_render', frame - 1)
        obj.hide_render = False
        insert_keyframe(obj, 'hide_render', frame)
        obj.hide_viewport = False
        insert_keyframe(obj, 'hide_viewport', frame)
    elif hasattr(obj, 'data'):
        if hasattr(obj.data, 'hide_render'):
            obj.data.hide_viewport = True
            insert_keyframe(obj, 'hide_viewport', frame - 1)
            obj.data.hide_render = True
            insert_keyframe(obj, 'hide_render', frame - 1)
            obj.data.hide_render = False
            insert_keyframe(obj, 'hide_render', frame)
            obj.data.hide_viewport = False
            insert_keyframe(obj, 'hide_viewport', frame)

    # for child in obj.children:
    #     unhide_frm_rec(child, frame)
    # obj.hide_set(True)


def hide_frm_rec(obj, frame):
    """
    hide b_object from render view until the time begin_time
    it works for mesh and curve objects

    :param obj:
    :param frame:
    :return:
    """
    obj = get_obj(obj)
    if hasattr(obj, 'hide_render'):
        obj.hide_render = False
        insert_keyframe(obj, 'hide_render', frame - 1)
        obj.hide_render = True
        insert_keyframe(obj, 'hide_render', frame)
    elif hasattr(obj, 'data'):
        if hasattr(obj.data, 'hide_rander'):
            obj.data.hide_render = False
            insert_keyframe(obj, 'hide_render', frame - 1)
            obj.data.hide_render = True
            insert_keyframe(obj, 'hide_render', frame)

    # for child in obj.children:
    #     hide_frm_rec(child, frame)
    # obj.hide_set(True)


'''
context.scene.
'''


def set_layout_clip_end(end=100000):
    for area in bpy.data.screens["Layout"].areas:
        if area.ui_type == 'VIEW_3D':
            for space in area.spaces:
                space.clip_end = end


def set_frame(frame):
    bpy.context.scene.frame_set(int(frame))


def get_frame():
    return bpy.context.scene.frame_current


def make_new_collection(name="MyCollection"):
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)


# linking and unlinking


def get_collection(name="Collection"):
    return bpy.data.collections.get(name)


def link(obj, collection=None):
    obj = get_obj(obj)

    # check, whether already linked object is added to the correct collection
    # (when a composed objected is added to a custom collection the children are automatically added to the default collection
    # A necessary relinking is initiated
    if len(obj.users_collection) > 0:
        old_collection = obj.users_collection[0].name
        if old_collection != collection:
            un_link(obj, old_collection)
            link(obj, collection)
    else:
        # unlinked object
        if collection is None:
            if obj.name not in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.link(obj)
        else:
            if collection not in bpy.context.scene.collection.children:
                make_new_collection(collection)
            bpy.context.scene.collection.children[collection].objects.link(obj)
            ### if there are children they need to be relinked if there is a custom collection
            for child in obj.children:
                link(child, collection=collection)


def recursive_link(obj, collection=None):
    """
        link a blender object to the collection with the name collection

        :param obj: object to link
        :param collection: name of the collection, the object is linked to
        :return:
        """
    obj = get_obj(obj)
    link(obj, collection)

    for child in obj.children:
        recursive_link(child, collection)


def un_link(obj, collection=None):
    """
        link a blender object to the collection with the name collection

        :param obj: object to link
        :param collection: name of the collection, the object is linked to
        :return:
        """
    if collection is None:
        bpy.context.scene.collection.objects.unlink(obj)
    else:
        bpy.context.scene.collection.children[collection].objects.unlink(obj)


def is_linked(obj):
    return obj.name in bpy.context.view_layer.objects


def delete(obj):
    for child in obj.children:
        delete(child)
    select(obj)
    bpy.ops.object.delete()


##################
# render engines #
##################

def set_render_engine(engine="CYCLES", transparent=False, motion_blur=False, denoising=False,
                      resolution_percentage=100, frame_start=0, taa_render_samples=1024, feature_set='SUPPORTED'):
    """
    @type frame_start: int
    
    """
    scene = get_scene()
    if engine == BLENDER_EEVEE:
        engine = BLENDER_EEVEE
    scene.render.engine = engine
    scene.render.use_compositing = True
    scene.render.resolution_percentage = resolution_percentage
    scene.render.film_transparent = transparent

    if engine == BLENDER_EEVEE:
        scene.eevee.use_gtao = True
        scene.eevee.use_bloom = True
        scene.render.use_motion_blur = motion_blur
        scene.eevee.use_ssr = True  # (space reflections)
        scene.eevee.use_ssr_halfres = False  # (space reflections)
        scene.eevee.use_ssr_refraction = True  # (space reflections)
        scene.eevee.ssr_quality = 1  # (space reflections)
        scene.eevee.ssr_max_roughness = 0  # (space reflections)
        scene.frame_start = frame_start
        set_taa_render_samples(taa_render_samples, begin_frame=0)

        # set view to Material view

        area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
        space = next(space for space in area.spaces if space.type == 'VIEW_3D')
        space.shading.type = 'MATERIAL'
    else:
        scene.cycles.feature_set = feature_set

        # set view to Material view

        area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
        space = next(space for space in area.spaces if space.type == 'VIEW_3D')
        space.shading.type = 'MATERIAL'  # set the viewport shading

    create_composition(denoising=denoising)


#######################
# work with materials #
#######################

def mix_color(bob, from_value, to_value, begin_frame, frame_duration):
    '''
    changes the value of the first mixer node
    :param from_value:
    :param bob:
    :param value:
    :return:
    '''
    obj = get_obj(bob)
    if obj.data and len(obj.data.materials) > 0:
        mat = obj.data.materials[0]
        nodes = mat.node_tree.nodes
        mixer = nodes['Mix']  # get first mixer

        factor = mixer.inputs['Fac']
        factor.default_value = from_value
        insert_keyframe(factor, 'default_value', begin_frame)
        factor.default_value = to_value
        insert_keyframe(factor, 'default_value', begin_frame + frame_duration)


def shader_value(bob, from_value, to_value, begin_frame, frame_duration):
    '''
    changes the value of the first value node in the material of the bob
    :param from_value:
    :param bob:
    :param value:
    :return:
    '''
    obj = get_obj(bob)
    if obj.data and len(obj.data.materials) > 0:
        mat = obj.data.materials[0]
        nodes = mat.node_tree.nodes
        v_node = nodes['Value']  # get first mixer

        value = v_node.outputs['Value']
        value.default_value = from_value
        insert_keyframe(value, 'default_value', begin_frame)
        value.default_value = to_value
        insert_keyframe(value, 'default_value', begin_frame + frame_duration)


def set_emission(material, emission):
    # find link to base color
    links = material.node_tree.links
    from_socket = None
    nodes = material.node_tree.nodes
    bsdf = nodes['Principled BSDF']
    if bsdf:
        link_exists = False
        for link in links:
            if link.to_node == bsdf:
                if link.to_socket == bsdf.inputs['Base Color']:
                    from_socket = link.from_socket
                if link.to_socket == bsdf.inputs[EMISSION]:
                    link_exists = True
        if from_socket and not link_exists:
            links.new(from_socket, bsdf.inputs[EMISSION])
        else:
            bsdf.inputs[EMISSION].default_value = bsdf.inputs['Base Color'].default_value
    bsdf.inputs['Emission Strength'].default_value = emission


def change_emission_to(bob, value, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
    obj = get_obj(bob)
    material = obj.data.materials[0]
    frame = begin_time * FRAME_RATE
    if material:
        nodes = material.node_tree.nodes
        bsdf = nodes['Principled BSDF']
        strength = bsdf.inputs['Emission Strength']
        if bsdf:
            set_frame(frame - 1)
            insert_keyframe(strength, 'default_value', frame)  # preserve old value up to the moment of change
            strength.default_value = value
            insert_keyframe(strength, 'default_value', frame + transition_time * FRAME_RATE)


def set_emission_color(bob, color):
    obj = get_obj(bob)
    material = obj.data.materials[0]
    if material:
        nodes = material.node_tree.nodes
        bsdf = nodes['Principled BSDF']
        emission = bsdf.inputs[EMISSION]
        index = COLOR_NAMES.index(color)
        col = COLORS[index]
        col = [i / 255 for i in col[0:3]] + [1]
        emission.default_value = deepcopy(col)
        bsdf.inputs['Emission Strength'].default_value = 0


def change_emission(bob, from_value=0, to_value=1, begin_frame=0, frame_duration=1):
    obj = get_obj(bob)
    if obj.data and obj.data.materials:
        material = obj.data.materials[0]
        nodes = material.node_tree.nodes
        if 'Principled BSDF' in nodes:
            bsdf = nodes['Principled BSDF']
            if bsdf.inputs[EMISSION].default_value[0:3] == (0, 0, 0):
                bsdf.inputs[EMISSION].default_value = bsdf.inputs['Base Color'].default_value
            bsdf.inputs['Emission Strength'].default_value = from_value
            insert_keyframe(bsdf.inputs['Emission Strength'], 'default_value', frame=begin_frame)
            bsdf.inputs['Emission Strength'].default_value = to_value
            insert_keyframe(bsdf.inputs['Emission Strength'], 'default_value', frame=begin_frame + frame_duration)


def change_emission_by_name(name_part, from_value, to_value, begin_frame, frame_duration):
    for mat in bpy.data.materials:
        if name_part in mat.name:
            nodes = mat.node_tree.nodes
            if 'Principled BSDF' in nodes:
                bsdf = nodes['Principled BSDF']
                if bsdf.inputs[EMISSION].default_value[0:3] == (0, 0, 0):
                    bsdf.inputs[EMISSION].default_value = bsdf.inputs['Base Color'].default_value
                bsdf.inputs['Emission Strength'].default_value = from_value
                insert_keyframe(bsdf.inputs['Emission Strength'], 'default_value', frame=begin_frame)
                bsdf.inputs['Emission Strength'].default_value = to_value
                insert_keyframe(bsdf.inputs['Emission Strength'], 'default_value', frame=begin_frame + frame_duration)


def change_alpha_by_name(name_part, from_value, to_value, begin_frame, frame_duration):
    for mat in bpy.data.materials:
        if name_part in mat.name:
            nodes = mat.node_tree.nodes
            if 'Principled BSDF' in nodes:
                bsdf = nodes['Principled BSDF']
                bsdf.inputs['Alpha'].default_value = from_value
                insert_keyframe(bsdf.inputs['Alpha'], 'default_value', frame=begin_frame)
                bsdf.inputs['Alpha'].default_value = to_value
                insert_keyframe(bsdf.inputs['Alpha'], 'default_value', frame=begin_frame + frame_duration)


def animate_glare(glare, **kwargs):
    begin_frame = get_from_kwargs(kwargs, 'begin_frame', 0)
    end_frame = get_from_kwargs(kwargs, 'end_frame', DEFAULT_ANIMATION_TIME * FRAME_RATE)

    threshold_start = get_from_kwargs(kwargs, 'threshold_start', 10)
    threshold_end = get_from_kwargs(kwargs, 'threshold_end', 0)

    glare.threshold = threshold_start
    insert_keyframe(glare, 'threshold', begin_frame)
    glare.threshold = threshold_end
    insert_keyframe(glare, 'threshold', end_frame)


def animate_sky_background(sky, **kwargs):
    begin_frame = get_from_kwargs(kwargs, 'begin_frame', 0)
    end_frame = get_from_kwargs(kwargs, 'end_frame', DEFAULT_ANIMATION_TIME * FRAME_RATE)

    elevation_start = get_from_kwargs(kwargs, 'elevation_start', 10)
    elevation_end = get_from_kwargs(kwargs, 'elevation_end', 0)

    rotation_start = get_from_kwargs(kwargs, 'rotation_start', 0)
    rotation_end = get_from_kwargs(kwargs, 'rotation_end', 45)

    sky.sun_elevation = elevation_start
    sky.sun_rotation = rotation_start
    insert_keyframe(sky, 'sun_elevation', begin_frame)
    insert_keyframe(sky, 'sun_rotation', begin_frame)
    sky.sun_elevation = elevation_end
    sky.sun_rotation = rotation_end
    insert_keyframe(sky, 'sun_elevation', end_frame)
    insert_keyframe(sky, 'sun_rotation', end_frame)


def set_hdri_background(filename='', ext='exr', simple=False, transparent=False, background="background", strength=0.2,
                        no_transmission_ray=False, rotation_euler=None):
    # remove lights
    for obj in bpy.data.objects:
        if 'Sun' in obj.name:
            un_link(obj, collection='Collection')
    world = bpy.data.worlds[-1]
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    out = nodes['World Output']
    light_path = nodes['Light Path']
    rgb = nodes['RGB']

    # nodes.remove(nodes['Mix'])
    nodes.remove(rgb)

    bg = nodes['Background']
    bg.location = (-400, 0)
    environment = nodes.new(type='ShaderNodeTexEnvironment')
    environment.location = (-700, 0)
    environment.image = bpy.data.images.load(RES_HDRI_DIR + "/" + filename + '.' + ext)

    if rotation_euler is not None:
        tex = nodes.new(type='ShaderNodeTexCoord')
        tex.location = (-1200, 0)
        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.location = (-1000, 0)
        mapping.vector_type = 'NORMAL'
        mapping.inputs['Rotation'].default_value = rotation_euler
        links.new(tex.outputs['Generated'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], environment.inputs['Vector'])

    if simple:
        # latest configuration first used for penrose video

        nodes.remove(light_path)
        nodes.remove(bg)
        links.new(environment.outputs['Color'], out.inputs['Surface'])
    else:

        max = nodes.new(type='ShaderNodeMath')
        max.location = (-400, 400)
        max.operation = 'MAXIMUM'

        if background == 'gradient':
            mix_shader = nodes.new(type='ShaderNodeMixShader')
            mix_shader.location = (-200, 0)
            links.new(mix_shader.outputs['Shader'], out.inputs['Surface'])
            links.new(environment.outputs['Color'], bg.inputs['Color'])
            links.new(bg.outputs['Background'], mix_shader.inputs[1])
            light_path.location = (-400, 500)

            if no_transmission_ray:
                links.new(light_path.outputs['Is Transmission Ray'], max.inputs[1])
                links.new(max.outputs['Value'], mix_shader.inputs['Fac'])
                links.new(light_path.outputs['Is Camera Ray'], max.inputs[0])
            else:
                links.new(light_path.outputs['Is Camera Ray'], mix_shader.inputs['Fac'])
                nodes.remove(max)

            bg2 = nodes.new(type='ShaderNodeBackground')
            bg2.location = (-400, -400)
            links.new(bg2.outputs['Background'], mix_shader.inputs[2])
            ramp = nodes.new(type='ShaderNodeValToRGB')
            ramp.location = (-700, -500)
            ramp.color_ramp.elements[0].color = [0.034, 0.047, 0.053, 1]
            ramp.color_ramp.elements[1].color = [0.147, 0.212, 0.262, 1]
            links.new(ramp.outputs['Color'], bg2.inputs['Color'])
            sep = nodes.new(type='ShaderNodeSeparateXYZ')
            sep.location = (-1000, -400)
            links.new(sep.outputs['Z'], ramp.inputs['Fac'])
            mapping2 = nodes.new(type='ShaderNodeMapping')
            mapping2.location = (-1300, -400)
            mapping2.inputs['Rotation'].default_value = [np.pi / 2, 0, 0]
            links.new(mapping2.outputs['Vector'], sep.inputs['Vector'])
            coords = nodes.new(type='ShaderNodeTexCoord')
            coords.location = (-1500, -400)
            links.new(coords.outputs['Window'], mapping2.inputs['Vector'])
        elif isinstance(background, str):
            mix_shader = nodes.new(type='ShaderNodeMixShader')
            mix_shader.location = (-200, 0)
            links.new(mix_shader.outputs['Shader'], out.inputs['Surface'])
            links.new(environment.outputs['Color'], bg.inputs['Color'])
            links.new(bg.outputs['Background'], mix_shader.inputs[1])
            light_path.location = (-400, 500)

            if no_transmission_ray:
                links.new(light_path.outputs['Is Transmission Ray'], max.inputs[1])
                links.new(max.outputs['Value'], mix_shader.inputs['Fac'])
                links.new(light_path.outputs['Is Camera Ray'], max.inputs[0])
            else:
                links.new(light_path.outputs['Is Camera Ray'], mix_shader.inputs['Fac'])
                nodes.remove(max)

            links.new(environment.outputs['Color'], bg.inputs['Color'])
            links.new(bg.outputs['Background'], mix_shader.inputs[1])
            bg2 = nodes.new(type='ShaderNodeBackground')
            bg2.location = (-400, -400)
            links.new(bg2.outputs['Background'], mix_shader.inputs[2])
            bg2.inputs['Color'].default_value = get_color(background)
            bg2.inputs['Strength'].default_value = strength
        else:
            # delete default nodes
            nodes.remove(light_path)
            links.new(bg.outputs['Background'], out.inputs['Surface'])
            links.new(environment.outputs['Color'], bg.inputs['Color'])

    bpy.data.scenes["Scene"].render.film_transparent = transparent


def hdri_background_rotate(rotation_euler=[0, 0, 0], begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
    world = bpy.data.worlds[-1]
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    mapping = nodes['Mapping']
    insert_keyframe(mapping.inputs['Rotation'], 'default_value', begin_time * FRAME_RATE)
    mapping.inputs['Rotation'].default_value = rotation_euler
    insert_keyframe(mapping.inputs['Rotation'], 'default_value', (begin_time + transition_time) * FRAME_RATE)
    return begin_time + transition_time


def set_hdri_strength(strength, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
    frame = int(begin_time * FRAME_RATE)

    nodes = bpy.data.worlds["World"].node_tree.nodes

    background_nodes = []
    for node in nodes:
        if 'Background' in node.name:  # only the strength of the hdri is increased
            background_nodes.append(node)
    for node in background_nodes:
        strength_socket = node.inputs['Strength']
        if frame > 0:
            set_frame(frame - 1)
            strength_old = strength_socket.default_value
        else:
            strength_old = 0
        strength_socket.default_value = strength_old
        insert_keyframe(strength_socket, 'default_value', frame - 1)
        strength_socket.default_value = strength
        insert_keyframe(strength_socket, 'default_value', frame + int(transition_time * FRAME_RATE))

    return begin_time + transition_time


def change_hdri_strength(start, end, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
    frame = int(begin_time * FRAME_RATE)
    strength_socket = bpy.data.worlds["World"].node_tree.nodes["Background.001"].inputs['Strength']

    strength_socket.default_value = start
    insert_keyframe(strength_socket, 'default_value', frame)
    strength_socket.default_value = end
    insert_keyframe(strength_socket, 'default_value', frame + int(transition_time * FRAME_RATE))


def dim_background(value=[0, 0, 0, 1], begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, transparent=True):
    print("Dim background to", value, "at the time", begin_time)
    nodes = bpy.data.worlds["World"].node_tree.nodes
    if 'Mix' in nodes:
        mixer = nodes["Mix"]

    insert_keyframe(mixer.inputs[1], 'default_value', frame=begin_time * FRAME_RATE)
    mixer.inputs[1].default_value = value
    insert_keyframe(mixer.inputs[1], 'default_value', frame=(begin_time + transition_time) * FRAME_RATE)

    bpy.data.scenes["Scene"].render.film_transparent = transparent


def get_color(color):
    if isinstance(color, list):
        if len(list)==4:
            return color
        elif len(list)==3:
            return list+[1]
        elif len(list)==1:
            return list*3+[1]
        else:
            return [1,1,1,1]
    else:
        return get_color_from_string(color)


def get_color_from_string(color_str):
    for prefix in COLOR_PREFIXES:
        if prefix in color_str:
            color_str = color_str[len(prefix) + 1:]
    color_index = COLOR_NAMES.index(color_str)
    if color_index > -1:
        return COLORS_SCALED[color_index]
    else:
        return [1, 1, 1, 1]


def get_new_material(name="new_material"):
    return bpy.data.materials.new(name=name)


def change_material_properties(bob, slot=0, begin_frame=0, frame_duration=1, **kwargs):
    obj = get_obj(bob)
    material = obj.material_slots[slot].material

    nodes = material.node_tree.nodes
    if 'Principled BSDF' in nodes:
        bsdf = nodes['Principled BSDF']
    else:
        bsdf = None

    if bsdf:
        if 'alpha' in kwargs:
            alpha = kwargs.pop('alpha')
            bsdf_alpha = bsdf.inputs['Alpha']
            bsdf_alpha.default_value = 0
            insert_keyframe(bsdf_alpha, "default_value", begin_frame)
            bsdf_alpha.default_value = 1
            insert_keyframe(bsdf_alpha, "default_value", begin_frame + frame_duration)


def set_alpha_for_material(material, alpha):
    """
    If you want to access alpha channel and want to have fade in at the same time
    you have to create an 'AlphaFactor' mixing node
    :param material:
    :param alpha:
    :return:
    """
    if material:
        alpha_node = None
        for n in material.node_tree.nodes:
            if 'AlphaFactor' in n.name or 'AlphaFactor' in n.label:
                alpha_node = n
                break
        if alpha_node:
            alpha_node.inputs[0].default_value = alpha
            return [alpha_node.inputs[0]]
        elif 'Principled BSDF' in material.node_tree.nodes:
            bsdf = material.node_tree.nodes['Principled BSDF']
            bsdf.inputs['Alpha'].default_value = alpha  # standard material
            return [bsdf.inputs['Alpha']]  # return for key_framing
        elif 'Mix Shader' in material.node_tree.nodes:
            mix1 = material.node_tree.nodes['Mix Shader']
            mix2 = material.node_tree.nodes['Mix Shader.001']
            if mix1 is None or mix2 is None:
                raise Warning("alpha cannot be set for non BSDF-Materials.")
            else:  # translucent material
                mix1.inputs[0].default_value = alpha
                mix2.inputs[0].default_value = alpha
                return [mix1.inputs[0], mix2.inputs[0]]  # return for key_framing


def set_metallic_for_material(material, metallic):
    bsdf = material.node_tree.nodes['Principled BSDF']
    if bsdf is None:
        raise Warning("metallic cannot be set for non BSDF-Materials.")
    bsdf.inputs['Metallic'].default_value = metallic


def set_roughness_for_material(material, roughness):
    bsdf = material.node_tree.nodes['Principled BSDF']
    if bsdf is None:
        raise Warning("roughness cannot be set for non BSDF-Materials.")
    bsdf.inputs['Roughness'].default_value = roughness


def set_emission_strength_for_material(material, emission_strength):
    bsdf = material.node_tree.nodes['Principled BSDF']
    if bsdf is None:
        raise Warning("emission strength cannot be set for non BSDF-Materials.")
    bsdf.inputs['Emission Strength'].default_value = emission_strength


def set_transmission_for_material(material, transmission):
    bsdf = material.node_tree.nodes['Principled BSDF']
    if bsdf is None:
        raise Warning("transmission cannot be set for non BSDF-Materials.")
    bsdf.inputs[TRANSMISSION].default_value = transmission


def get_alpha_at_current_keyframe(obj, frame, slot=0):
    if len(obj.material_slots) > 0:
        current_frame = get_frame()
        set_frame(frame)
        material_slot = obj.material_slots[slot]
        material = material_slot.material
        if _alpha := material.node_tree.nodes.get("AlphaFactor"):
            alpha = _alpha.inputs[0].default_value
        # Detect AlphaFactor node (book and pages for instance)
        elif 'Principled BSDF' in material.node_tree.nodes:
            alpha = material.node_tree.nodes["Principled BSDF"].inputs['Alpha'].default_value
        else:
            alpha = material.node_tree.nodes['Mix Shader'].inputs[0].default_value
        set_frame(current_frame)
        return alpha
    return 0


def set_alpha_and_keyframe(obj, value, frame, offset_for_slots=None):
    """
    keyframe the alpha value of an object
    :param offset_for_slots:
    :param obj:
    :param value:
    :param frame:
    :return:
    """
    obj = get_obj(obj)

    for s, material_slot in enumerate(obj.material_slots):
        material = material_slot.material
        dialers = set_alpha_for_material(material, value)
        if offset_for_slots is not None and len(offset_for_slots) > s:
            offset = offset_for_slots[s] * FRAME_RATE
        else:
            offset = 0
        if dialers:
            for dialer in dialers:
                insert_keyframe(dialer, 'default_value', frame + offset)


# Geometry nodes

def get_node_tree(name, type='GeometryNodeTree'):
    return bpy.data.node_groups.new(name, type='GeometryNodeTree')


# Materials/Shaders

def customize_material(material, **kwargs):
    """
    if you want to override special materials use override_material flag
    :param material:
    :param kwargs:
    :return:
    """

    # for nice alpha transitions in EEVEE
    material.blend_method = 'HASHED'
    material.shadow_method = 'HASHED'

    override_material = get_from_kwargs(kwargs, 'override_material', True)

    if 'emission' in kwargs:
        emission = kwargs.pop('emission')
    else:
        emission = None

    if 'transmission' in kwargs:
        transmission = kwargs.pop('transmission')
    else:
        transmission = None

    if 'roughness' in kwargs:
        roughness = kwargs.pop('roughness')
    else:
        roughness = None

    if 'ior' in kwargs:  # refraction index
        ior = kwargs.pop('ior')
    else:
        ior = None

    if 'metallic' in kwargs:  # refraction index
        metallic = kwargs.pop('metallic')
    else:
        metallic = None

    if 'brightness' in kwargs:  # brightness of color: all values are rescaled such that the strongest component has the value brightness
        brightness = kwargs.pop('brightness')
    else:
        brightness = None

    if 'scatter' in kwargs:
        scatter = kwargs.pop('scatter')
    else:
        scatter = 0

    if 'specular_tint' in kwargs:
        specular_tint = kwargs.pop('specular_tint')
    else:
        specular_tint = None

    if 'alpha' in kwargs:
        alpha = kwargs.pop('alpha')
    else:
        alpha = 1

    if 'Principled BSDF' in material.node_tree.nodes:
        bsdf = material.node_tree.nodes['Principled BSDF']
    else:
        bsdf = None
        if emission is not None and emission > 0:
            if 'Emission' in material.node_tree.nodes:
                emission_node = material.node_tree.nodes['Emission']
                emission_node.inputs['Strength'].default_value = emission

    if bsdf:
        special_material = False
        for special in SPECIALS:
            if special in material.name:
                special_material = True
                break
        if not special_material or override_material:

            if transmission is not None:
                bsdf.inputs[TRANSMISSION].default_value = transmission
            if roughness is not None:
                bsdf.inputs['Roughness'].default_value = roughness
            if ior is not None:
                bsdf.inputs['IOR'].default_value = ior
            if metallic is not None:
                bsdf.inputs['Metallic'].default_value = metallic
            if specular_tint:
                bsdf.inputs['Specular Tint'].default_value = specular_tint

            if brightness is not None:
                [red, green, blue, alpha] = material.node_tree.nodes['Principled BSDF'].inputs[
                                                'Base Color'].default_value[0:4]

                maximum = np.max([red, green, blue])
                scale = brightness / maximum

                red1 = red * scale
                green1 = green * scale
                blue1 = blue * scale
                material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = [red1, green1, blue1,
                                                                                                  alpha]

        if emission is not None:
            set_emission(material, emission)

        bsdf.inputs['Alpha'].default_value = alpha  # this get's overriden during fade_in unfortunately

    if scatter > 0:
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        mat_out = nodes['Material Output']
        vol_scatter = nodes.new(type='ShaderNodeVolumeScatter')
        vol_scatter.inputs['Density'].default_value = scatter
        links.new(vol_scatter.outputs['Volume'], mat_out.inputs['Volume'])

    return material


def get_material(material, **kwargs):
    if isinstance(material, str):
        if material == 'image' and 'src' in kwargs:
            return make_image_material(kwargs['src']).copy()
        elif material == 'gradient':
            material = make_gradient_material(**kwargs)
        elif material == 'dashed':
            material = make_dashed_material(**kwargs)
        elif material == 'magnet':
            material = make_magnet_material(**kwargs)
        elif material == 'mandelbrot':
            material = make_mandelbrot_material(**kwargs)
        elif material == 'iteration':
            material = make_iteration_material(**kwargs)
        elif material == 'hue':
            material = make_hue_material(**kwargs)
        else:
            material = bpy.data.materials[material].copy()
        # override default values
        material = customize_material(material, **kwargs)

        return material


def make_magnet_material(**kwargs):
    direction = get_from_kwargs(kwargs, 'direction', 'X')
    material = bpy.data.materials.new(name="magnet_" + direction)
    # for eevee
    material.use_screen_refraction = True
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    out = nodes.get("Material Output")

    # frenel and transparent bsdf

    mix_shader = nodes.new(type='ShaderNodeMixShader')
    mix_shader.location = (200, 800)
    links.new(mix_shader.outputs['Shader'], out.inputs['Surface'])
    links.new(bsdf.outputs['BSDF'], mix_shader.inputs[2])

    trans_bsdf = nodes.new(type='ShaderNodeBsdfTransparent')
    trans_bsdf.location = (0, 400)
    links.new(trans_bsdf.outputs['BSDF'], mix_shader.inputs[1])

    abs2 = nodes.new(type='ShaderNodeMath')
    abs2.operation = 'ABSOLUTE'
    abs2.location = (0, 600)
    links.new(abs2.outputs['Value'], mix_shader.inputs['Fac'])

    # first mixer
    mixer = nodes.new(type='ShaderNodeMixRGB')
    mixer.location = (-200, 0)
    mixer.inputs['Color1'].default_value = [1, 1, 1, 1]
    links.new(mixer.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(mixer.outputs['Color'], bsdf.inputs[EMISSION])

    # two more mixers and range node

    mixer2 = nodes.new(type='ShaderNodeMixRGB')
    mixer2.location = (-400, -200)
    mixer2.inputs['Color1'].default_value = [1, 1, 1, 1]
    links.new(mixer2.outputs['Color'], mixer.inputs['Color2'])

    mixer3 = nodes.new(type='ShaderNodeMixRGB')
    mixer3.location = (-400, +200)
    mixer3.inputs['Color1'].default_value = [1, 1, 1, 1]
    links.new(mixer3.outputs['Color'], mixer.inputs['Color1'])

    range = nodes.new(type='ShaderNodeMapRange')
    range.location = (-400, 500)
    range.inputs['From Min'].default_value = -1
    range.inputs['From Max'].default_value = 1
    range.inputs['To Min'].default_value = 0
    range.inputs['To Max'].default_value = 1
    links.new(range.outputs['Result'], mixer.inputs['Fac'])

    # two math nodes

    sign = nodes.new(type='ShaderNodeMath')
    sign.location = (-600, 500)
    sign.operation = 'SIGN'
    links.new(sign.outputs['Value'], range.inputs['Value'])

    abs = nodes.new(type='ShaderNodeMath')
    abs.location = (-600, 300)
    abs.operation = 'ABSOLUTE'
    links.new(abs.outputs['Value'], mixer2.inputs['Fac'])
    links.new(abs.outputs['Value'], mixer3.inputs['Fac'])

    # value node and two ramps
    val = nodes.new(type='ShaderNodeValue')
    val.location = (-800, 400)
    links.new(val.outputs['Value'], sign.inputs['Value'])
    links.new(val.outputs['Value'], abs.inputs['Value'])
    links.new(val.outputs['Value'], abs2.inputs['Value'])

    ramp = nodes.new(type='ShaderNodeValToRGB')
    ramp.location = (-800, 0)
    ramp.color_ramp.elements[0].position = 0
    ramp.color_ramp.elements[0].color = [0, 1, 0, 1]
    ramp.color_ramp.elements[1].position = 0.81
    ramp.color_ramp.elements[1].color = [1, 0, 0, 1]
    links.new(ramp.outputs['Color'], mixer2.inputs['Color2'])

    ramp2 = nodes.new(type='ShaderNodeValToRGB')
    ramp2.location = (-800, -400)
    ramp2.color_ramp.elements[0].position = 0.19
    ramp2.color_ramp.elements[0].color = [1, 0, 0, 1]
    ramp2.color_ramp.elements[1].position = 1
    ramp2.color_ramp.elements[1].color = [0, 1, 0, 1]
    links.new(ramp2.outputs['Color'], mixer3.inputs['Color2'])

    sep_xyz = nodes.new(type='ShaderNodeSeparateXYZ')
    sep_xyz.location = (-1000, 0)
    if direction == "X":
        links.new(sep_xyz.outputs['X'], ramp.inputs['Fac'])
        links.new(sep_xyz.outputs['X'], ramp2.inputs['Fac'])
    elif direction == 'Y':
        links.new(sep_xyz.outputs['Y'], ramp.inputs['Fac'])
        links.new(sep_xyz.outputs['Y'], ramp2.inputs['Fac'])
    else:
        links.new(sep_xyz.outputs['Z'], ramp.inputs['Fac'])
        links.new(sep_xyz.outputs['Z'], ramp2.inputs['Fac'])
    coords = nodes.new(type='ShaderNodeTexCoord')
    coords.location = (-1200, 0)
    links.new(coords.outputs['Generated'], sep_xyz.inputs['Vector'])
    return material


def make_coarse_grained_group(name='CoarseGraining'):
    cg = bpy.data.node_groups.new(name, 'ShaderNodeTree')
    nodes = cg.nodes
    links = cg.links

    left = -1000
    width = 200

    group_inputs = nodes.new('NodeGroupInput')
    group_inputs.location = (left, 0)

    group_outputs = nodes.new('NodeGroupOutput')
    group_outputs.location = (0, 0)

    make_new_socket(cg, name='Value', io='INPUT', type='NodeSocketFloat')
    make_new_socket(cg, name='spacing', io='INPUT', type='NodeSocketFloat')
    make_new_socket(cg, name='Value', io='OUTPUT', type='NodeSocketFloat')
    # cg.inputs.new('NodeSocketFloat', 'Value')
    # cg.inputs.new('NodeSocketFloat', 'spacing')
    # cg.outputs.new('NodeSocketFloat', 'Value')

    mul = nodes.new(type='ShaderNodeMath')
    mul.operation = 'DIVIDE'
    mul.location = (left + width, 0)
    links.new(group_inputs.outputs[0], mul.inputs[0])
    links.new(group_inputs.outputs[1], mul.inputs[1])

    floor = nodes.new(type='ShaderNodeMath')
    floor.operation = 'FLOOR'
    floor.location = (left + 2 * width, 0)
    links.new(mul.outputs[0], floor.inputs[0])

    div = nodes.new(type='ShaderNodeMath')
    div.operation = 'MULTIPLY'
    div.location = (left + 3 * width, 0)
    links.new(floor.outputs[0], div.inputs[0])
    links.new(group_inputs.outputs[1], div.inputs[1])
    links.new(div.outputs[0], group_outputs.inputs[0])

    return cg


def make_phase2color_group(name='Phase2Color', threshold=2):
    p2c_tree = bpy.data.node_groups.new(name, 'ShaderNodeTree')

    group_nodes = p2c_tree.nodes
    grp_links = p2c_tree.links

    left = -1000
    width = 200

    group_inputs = group_nodes.new('NodeGroupInput')
    group_inputs.location = (left, 0)

    group_outputs = group_nodes.new('NodeGroupOutput')
    group_outputs.location = (0, 0)

    make_new_socket(p2c_tree, name='Complex', io='INPUT', type='NodeSocketVector')
    make_new_socket(p2c_tree, name='Color', io='OUTPUT', type='NodeSocketColor')
    # p2c_tree.inputs.new('NodeSocketVector', 'Complex')
    # p2c_tree.outputs.new('NodeSocketColor', 'Color')

    sep_xyz = group_nodes.new(type='ShaderNodeSeparateXYZ')
    sep_xyz.location = (left + width, 0)

    grp_links.new(group_inputs.outputs['Complex'], sep_xyz.inputs['Vector'])

    atan = group_nodes.new(type='ShaderNodeMath')
    atan.operation = 'ARCTAN2'
    atan.location = (left + 2 * width, 0)
    grp_links.new(sep_xyz.outputs['X'], atan.inputs[1])
    grp_links.new(sep_xyz.outputs['Y'], atan.inputs[0])

    phase_rescale = group_nodes.new(type='ShaderNodeMath')
    phase_rescale.location = (left + 3 * width, 0)
    phase_rescale.operation = 'MULTIPLY_ADD'
    phase_rescale.inputs[1].default_value = 0.5 / np.pi
    phase_rescale.inputs[2].default_value = 0.5
    grp_links.new(atan.outputs['Value'], phase_rescale.inputs[0])

    hue = group_nodes.new(type='ShaderNodeHueSaturation')
    hue.location = (left + 4 * width, 0)
    hue.inputs['Color'].default_value = [1, 0, 0, 1]

    # compute cutoff
    #
    # combine = group_nodes.new(type='ShaderNodeCombineXYZ')
    # combine.location = (left +  width, -400)
    # grp_links.new(sep_xyz.outputs['X'], combine.inputs['X'])
    # grp_links.new(sep_xyz.outputs['Y'], combine.inputs['Y'])
    #
    # length = group_nodes.new(type='ShaderNodeVectorMath')
    # length.operation='LENGTH'
    # length.location=(left+2*width,-400)
    # grp_links.new(combine.outputs['Vector'],length.inputs['Vector'])
    #
    # less = group_nodes.new(type='ShaderNodeMath')
    # less.operation='LESS_THAN'
    # less.location=(left+2*width,-400)
    # less.inputs[1].default_value=threshold
    # grp_links.new(length.outputs['Value'],less.inputs[0])
    #
    # grp_links.new(less.outputs['Value'],hue.inputs['Saturation'])
    # grp_links.new(less.outputs['Value'],hue.inputs['Value'])
    #
    grp_links.new(phase_rescale.outputs['Value'], hue.inputs['Hue'])
    grp_links.new(hue.outputs['Color'], group_outputs.inputs['Color'])

    return p2c_tree


def make_vector_interpolation(name='VectorInterpol', clamping_threshold=1000000):
    """
    The clamping threshold is necessary that the interpolation can take place even
    if some components are infinite. They are clamped between plus or minus the threshold

    :param name:
    :param clamping_threshold:
    :return:
    """
    interpol_tree = bpy.data.node_groups.new(name, 'ShaderNodeTree')

    group_nodes = interpol_tree.nodes
    grp_links = interpol_tree.links

    left = -1600
    width = 200

    group_inputs = group_nodes.new('NodeGroupInput')
    group_inputs.location = (left, 0)

    group_outputs = group_nodes.new('NodeGroupOutput')
    group_outputs.location = (0, 0)

    make_new_socket(interpol_tree, name='Initial', io='INPUT', type='NodeSocketVector')
    make_new_socket(interpol_tree, name='n_initial', io='INPUT', type='NodeSocketFloat')
    make_new_socket(interpol_tree, name='Final', io='INPUT', type='NodeSocketVector')
    make_new_socket(interpol_tree, name='n_final', io='INPUT', type='NodeSocketFloat')
    make_new_socket(interpol_tree, name='Factor', io='INPUT', type='NodeSocketFloat')
    make_new_socket(interpol_tree, name='Vector', io='OUTPUT', type='NodeSocketVector')
    make_new_socket(interpol_tree, name='n_out', io='OUTPUT', type='NodeSocketFloat')
    # interpol_tree.inputs.new('NodeSocketVector', 'Initial')
    # interpol_tree.inputs.new('NodeSocketVector', 'Final')
    # interpol_tree.inputs.new('NodeSocketFloat', 'n_initial')
    # interpol_tree.inputs.new('NodeSocketFloat', 'n_final')
    # interpol_tree.inputs.new('NodeSocketFloat', 'Factor')
    # interpol_tree.outputs.new('NodeSocketVector', 'Vector')
    # interpol_tree.outputs.new('NodeSocketFloat', 'n_out')

    # clamping Initial
    sep_xyz = group_nodes.new(type="ShaderNodeSeparateXYZ")
    sep_xyz.location = (left + width, 600)
    grp_links.new(group_inputs.outputs['Initial'], sep_xyz.inputs['Vector'])

    clamp1x = group_nodes.new(type='ShaderNodeClamp')
    clamp1x.location = (left + 2 * width, 900)
    clamp1x.inputs['Min'].default_value = -clamping_threshold
    clamp1x.inputs['Max'].default_value = clamping_threshold
    grp_links.new(sep_xyz.outputs['X'], clamp1x.inputs['Value'])

    clamp1y = group_nodes.new(type='ShaderNodeClamp')
    clamp1y.location = (left + 2 * width, 700)
    clamp1y.inputs['Min'].default_value = -clamping_threshold
    clamp1y.inputs['Max'].default_value = clamping_threshold
    grp_links.new(sep_xyz.outputs['Y'], clamp1y.inputs['Value'])

    clamp1z = group_nodes.new(type='ShaderNodeClamp')
    clamp1z.location = (left + 2 * width, 500)
    clamp1z.inputs['Min'].default_value = -clamping_threshold
    clamp1z.inputs['Max'].default_value = clamping_threshold
    grp_links.new(sep_xyz.outputs['Z'], clamp1z.inputs['Value'])

    combine1 = group_nodes.new(type='ShaderNodeCombineXYZ')
    combine1.location = (left + 3 * width, 600)
    grp_links.new(clamp1x.outputs['Result'], combine1.inputs['X'])
    grp_links.new(clamp1y.outputs['Result'], combine1.inputs['Y'])
    grp_links.new(clamp1z.outputs['Result'], combine1.inputs['Z'])

    # clamping Final

    sep_xyz2 = group_nodes.new(type="ShaderNodeSeparateXYZ")
    sep_xyz2.location = (left + width, 100)
    grp_links.new(group_inputs.outputs['Final'], sep_xyz2.inputs['Vector'])

    clamp2x = group_nodes.new(type='ShaderNodeClamp')
    clamp2x.location = (left + 2 * width, 300)
    clamp2x.inputs['Min'].default_value = -clamping_threshold
    clamp2x.inputs['Max'].default_value = clamping_threshold
    grp_links.new(sep_xyz2.outputs['X'], clamp2x.inputs['Value'])

    clamp2y = group_nodes.new(type='ShaderNodeClamp')
    clamp2y.location = (left + 2 * width, 100)
    clamp2y.inputs['Min'].default_value = -clamping_threshold
    clamp2y.inputs['Max'].default_value = clamping_threshold
    grp_links.new(sep_xyz2.outputs['Y'], clamp2y.inputs['Value'])

    clamp2z = group_nodes.new(type='ShaderNodeClamp')
    clamp2z.location = (left + 2 * width, -100)
    clamp2z.inputs['Min'].default_value = -clamping_threshold
    clamp2z.inputs['Max'].default_value = clamping_threshold
    grp_links.new(sep_xyz2.outputs['Z'], clamp2z.inputs['Value'])

    combine2 = group_nodes.new(type='ShaderNodeCombineXYZ')
    combine2.location = (left + 3 * width, 0)
    grp_links.new(clamp2x.outputs['Result'], combine2.inputs['X'])
    grp_links.new(clamp2y.outputs['Result'], combine2.inputs['Y'])
    grp_links.new(clamp2z.outputs['Result'], combine2.inputs['Z'])

    part1 = group_nodes.new(type='ShaderNodeVectorMath')
    part1.operation = 'SCALE'
    part1.location = (left + 4 * width, -300)
    grp_links.new(combine2.outputs['Vector'], part1.inputs['Vector'])
    grp_links.new(group_inputs.outputs['Factor'], part1.inputs['Scale'])

    scale2 = group_nodes.new(type='ShaderNodeMath')
    scale2.location = (left + 4 * width, 300)
    scale2.operation = 'SUBTRACT'
    scale2.inputs[0].default_value = 1
    grp_links.new(group_inputs.outputs['Factor'], scale2.inputs[1])

    part2 = group_nodes.new(type='ShaderNodeVectorMath')
    part2.operation = 'SCALE'
    part2.location = (left + 5 * width, 300)
    grp_links.new(combine1.outputs['Vector'], part2.inputs['Vector'])
    grp_links.new(scale2.outputs['Value'], part2.inputs['Scale'])

    sum = group_nodes.new(type='ShaderNodeVectorMath')
    sum.operation = 'ADD'
    sum.location = (left + 6 * width, 0)
    grp_links.new(part1.outputs['Vector'], sum.inputs[0])
    grp_links.new(part2.outputs['Vector'], sum.inputs[1])

    # mixing n
    delta = group_nodes.new(type="ShaderNodeMath")
    delta.operation = 'SUBTRACT'
    delta.location = (left + width, -300)
    delta.hide = True

    grp_links.new(group_inputs.outputs['n_initial'], delta.inputs[1])
    grp_links.new(group_inputs.outputs['n_final'], delta.inputs[0])

    multi_add = group_nodes.new(type='ShaderNodeMath')
    multi_add.operation = 'MULTIPLY_ADD'
    multi_add.location = (left + 2 * width, -400)

    grp_links.new(group_inputs.outputs['Factor'], multi_add.inputs[1])
    grp_links.new(group_inputs.outputs['n_initial'], multi_add.inputs[2])
    grp_links.new(delta.outputs['Value'], multi_add.inputs[0])

    grp_links.new(sum.outputs['Vector'], group_outputs.inputs['Vector'])
    grp_links.new(multi_add.outputs['Value'], group_outputs.inputs['n_out'])
    return interpol_tree


def make_mandel_iteration(name='Iteration', **kwargs):
    if 'threshold' in kwargs:
        threshold = kwargs.pop('threshold')
    else:
        threshold = 2

    p2c_tree = bpy.data.node_groups.new(name, 'ShaderNodeTree')

    group_nodes = p2c_tree.nodes
    grp_links = p2c_tree.links

    left = -2000
    width = 200

    group_inputs = group_nodes.new('NodeGroupInput')
    group_inputs.location = (left, 0)

    group_outputs = group_nodes.new('NodeGroupOutput')
    group_outputs.location = (0, 0)

    make_new_socket(p2c_tree, name='z', io='INPUT', type='NodeSocketVector')
    make_new_socket(p2c_tree, name='c', io='INPUT', type='NodeSocketVector')
    make_new_socket(p2c_tree, name='n', io='INPUT', type='NodeSocketFloat')
    # p2c_tree.inputs.new('NodeSocketVector', 'z')
    # p2c_tree.inputs.new('NodeSocketVector', 'c')
    # p2c_tree.inputs.new('NodeSocketFloat', 'n')
    make_new_socket(p2c_tree, name='z', io='OUTPUT', type='NodeSocketVector')
    make_new_socket(p2c_tree, name='n', io='OUTPUT', type='NodeSocketFloat')
    # p2c_tree.outputs.new('NodeSocketVector', 'z')
    # p2c_tree.outputs.new('NodeSocketFloat', 'n')

    sep_z = group_nodes.new(type='ShaderNodeSeparateXYZ')
    sep_z.location = (left + width, 300)
    grp_links.new(group_inputs.outputs['z'], sep_z.inputs['Vector'])

    sep_c = group_nodes.new(type='ShaderNodeSeparateXYZ')
    sep_c.location = (left + width, -600)
    grp_links.new(group_inputs.outputs['c'], sep_c.inputs['Vector'])

    # calculate interation
    # real part
    x2 = group_nodes.new(type='ShaderNodeMath')
    x2.operation = 'MULTIPLY'
    x2.location = (left + 2 * width, 500)
    grp_links.new(sep_z.outputs['X'], x2.inputs[0])
    grp_links.new(sep_z.outputs['X'], x2.inputs[1])

    y2 = group_nodes.new(type='ShaderNodeMath')
    y2.operation = 'MULTIPLY'
    y2.location = (left + 2 * width, 200)
    grp_links.new(sep_z.outputs['Y'], y2.inputs[0])
    grp_links.new(sep_z.outputs['Y'], y2.inputs[1])

    x2my2 = group_nodes.new(type='ShaderNodeMath')
    x2my2.operation = 'SUBTRACT'
    x2my2.location = (left + 3 * width, 300)
    grp_links.new(x2.outputs['Value'], x2my2.inputs[0])
    grp_links.new(y2.outputs['Value'], x2my2.inputs[1])

    x2my2pcx = group_nodes.new(type='ShaderNodeMath')
    x2my2pcx.operation = 'ADD'
    x2my2pcx.location = (left + 4 * width, 300)
    grp_links.new(x2my2.outputs['Value'], x2my2pcx.inputs[0])
    grp_links.new(sep_c.outputs['X'], x2my2pcx.inputs[1])

    # imaginary part
    xy = group_nodes.new(type='ShaderNodeMath')
    xy.operation = 'MULTIPLY'
    xy.location = (left + 2 * width, -300)
    grp_links.new(sep_z.outputs['X'], xy.inputs[0])
    grp_links.new(sep_z.outputs['Y'], xy.inputs[1])

    xy2pcy = group_nodes.new(type='ShaderNodeMath')
    xy2pcy.operation = 'MULTIPLY_ADD'
    xy2pcy.location = (left + 3 * width, -300)
    xy2pcy.inputs[1].default_value = 2
    grp_links.new(xy.outputs['Value'], xy2pcy.inputs[0])
    grp_links.new(sep_c.outputs['Y'], xy2pcy.inputs[2])

    combine = group_nodes.new(type='ShaderNodeCombineXYZ')
    combine.location = (left + 9 * width, 0)
    grp_links.new(x2my2pcx.outputs['Value'], combine.inputs['X'])
    grp_links.new(xy2pcy.outputs['Value'], combine.inputs['Y'])

    # count iterations below threshold
    combine2 = group_nodes.new(type='ShaderNodeCombineXYZ')
    combine2.location = (left + 5 * width, 0)
    grp_links.new(x2my2pcx.outputs['Value'], combine2.inputs['X'])
    grp_links.new(xy2pcy.outputs['Value'], combine2.inputs['Y'])
    # grp_links.new(sep_z.outputs['X'], combine2.inputs['X'])
    # grp_links.new(sep_z.outputs['Y'], combine2.inputs['Y'])

    length = group_nodes.new(type='ShaderNodeVectorMath')
    length.operation = 'LENGTH'
    length.location = (left + 6 * width, 0)
    grp_links.new(combine2.outputs['Vector'], length.inputs['Vector'])

    less = group_nodes.new(type='ShaderNodeMath')
    less.operation = 'LESS_THAN'
    less.location = (left + 7 * width, 0)
    less.inputs[1].default_value = threshold
    grp_links.new(length.outputs['Value'], less.inputs[0])

    addZ = group_nodes.new(type='ShaderNodeMath')
    addZ.operation = 'ADD'
    addZ.location = (left + 8 * width, 200)
    grp_links.new(sep_z.outputs['Z'], addZ.inputs[0])
    grp_links.new(less.outputs[0], addZ.inputs[1])
    grp_links.new(addZ.outputs['Value'], combine.inputs['Z'])

    grp_links.new(combine.outputs['Vector'], group_outputs.inputs['z'])

    # increase iteration counter
    addN = group_nodes.new(type='ShaderNodeMath')
    addN.operation = 'ADD'
    addN.location = (left + width, 0)
    addN.hide = True
    addN.inputs[0].default_value = 1

    grp_links.new(group_inputs.outputs['n'], addN.inputs[1])
    grp_links.new(addN.outputs['Value'], group_outputs.inputs['n'])
    return p2c_tree


def get_node_with_label(bob, label):
    mat = get_material_of(bob)
    for n in mat.node_tree.nodes:
        if n.label == label:
            return n


def get_geometry_node_from_modifier(modifier, label):
    for n in modifier.nodes:
        if label in n.label or label in n.name:
            return n
    return None


def get_material_from_modifier(modifier, label):
    for n in modifier.nodes:
        if label in n.label or label in n.name:
            if 'Material' in n.inputs:
                return n.inputs['Material'].default_value
    return None


def get_spacing_value(bob):
    mat = get_material_of(bob)
    for n in mat.node_tree.nodes:
        if n.label == 'Spacing':
            return n


def get_hue_value(bob):
    mat = get_material_of(bob)
    for n in mat.node_tree.nodes:
        if n.label == 'hue_node':
            return n


def make_alpha_frame(material, thickness):
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    if 'Texture Coordinate' in nodes:
        uv = nodes.get('Texture Coordinate')
    else:
        uv = nodes.new(type='ShaderNodeTexCoord')
        uv.location = (-1000, 0)

    bsdf = nodes.get('Principled BSDF')
    frame_group = create_group_from_vector_function(nodes,
                                                    ["t_x,x,<,x,1,t_x,-,<,+,2,/,t_y,y,<,y,1,t_y,-,<,+,2,/,+,2,/"],
                                                    parameters=['t'], name='frameGroup')
    frame_group.location = (-600, 0)

    if isinstance(thickness, float):
        thk = [thickness] * 3
    else:
        thk = thickness

    combine = nodes.new(type='ShaderNodeCombineXYZ')
    combine.location = (-800, 400)

    for i in range(3):
        combine.inputs[i].default_value = thk[i]

    length = nodes.new(type='ShaderNodeVectorMath')
    length.location = (-400, 0)
    length.operation = 'LENGTH'

    invert = nodes.new(type='ShaderNodeMath')
    invert.operation = 'SUBTRACT'
    invert.location = (-200, 0)
    invert.inputs[0].default_value = 1

    links.new(frame_group.outputs[0], length.inputs[0])
    links.new(combine.outputs[0], frame_group.inputs['t'])
    links.new(uv.outputs['UV'], frame_group.inputs[0])

    links.new(length.outputs['Value'], invert.inputs[1])
    links.new(invert.outputs[0], bsdf.inputs[TRANSMISSION])


def make_mandelbrot_material(**kwargs):
    if 'iterations' in kwargs:
        iterations = kwargs.pop('iterations')

    if 'phase' in kwargs:
        phase = kwargs.pop('phase')
    else:
        phase = False
    if 'threshold' in kwargs:
        threshold = kwargs.pop('threshold')
    else:
        threshold = 2

    if 'coarse_grained' in kwargs:
        coarse_grained = kwargs.pop('coarse_grained')
    else:
        coarse_grained = False

    material = bpy.data.materials.new(name="mandelbrot")
    # for eevee
    material.use_screen_refraction = True
    # for cycles
    material.cycles.displacement_method = 'DISPLACEMENT'
    material.use_nodes = True
    tree = material.node_tree
    nodes = material.node_tree.nodes
    links = tree.links
    bsdf = nodes.get("Principled BSDF")
    out = nodes.get("Material Output")

    outer_left = -1600

    uv = nodes.new(type='ShaderNodeTexCoord')
    uv.location = (outer_left - 1000, 400)

    old_iteration = None
    old_mixer = None

    if isinstance(iterations, range):
        interpolation = True
        iterations = list(iterations)[-1]
    elif isinstance(iterations, list):
        interpolation = True
        iterations = max(iterations)
    else:
        interpolation = False

    iteration_node = make_mandel_iteration(threshold=2)

    if interpolation:
        mixer = nodes.new(type='ShaderNodeGroup')
        mixer.node_tree = make_vector_interpolation('VectorInterpol')
        mixer.name = 'VectorInterpol'  # it's important to set the name to access the interpolation factor later on
        mixer.location = (-600, 0)
        mixer.hide = True
        mixer.inputs['Factor'].default_value = 0

        # add offset for z-component to have no cutoff initially
        add_vector = nodes.new(type='ShaderNodeVectorMath')
        add_vector.location = (-600, 300)
        add_vector.inputs[1].default_value = [0, 0, 1]
        add_vector.hide = True

        links.new(uv.outputs['Object'], add_vector.inputs[0])
        links.new(add_vector.outputs['Vector'], mixer.inputs['Initial'])
        old_mixer = None

    if coarse_grained:
        coarse_grained = nodes.new(type='ShaderNodeGroup')
        coarse_grained.node_tree = make_coarse_grained_group()
        coarse_grained.location = (outer_left - 300, 400)
        coarse_grained.hide = True
        coarse_grained2 = nodes.new(type='ShaderNodeGroup')
        coarse_grained2.location = (outer_left - 300, 500)
        coarse_grained2.node_tree = make_coarse_grained_group()
        coarse_grained2.hide = True

        grid_shader = create_shader_group_from_function(nodes,
                                                        "x,spacing,/,round,spacing,*,x,-,abs,thickness,<,y,spacing,/,round,spacing,*,y,-,abs,thickness,<,+",
                                                        ["spacing", "thickness"], "VECTOR", "FLOAT", "gridGroup")
        grid_shader.location = (outer_left - 300, 1600)
        links.new(uv.outputs['Object'], grid_shader.inputs['In'])

        sep = nodes.new(type='ShaderNodeSeparateXYZ')
        sep.location = (outer_left - 800, 400)
        links.new(sep.outputs['X'], coarse_grained.inputs[0])
        links.new(sep.outputs['Y'], coarse_grained2.inputs[0])

        val = nodes.new(type='ShaderNodeValue')
        val.label = 'Spacing'
        val.outputs[0].default_value = 1
        val.location = (outer_left - 400, 1600)
        links.new(val.outputs[0], coarse_grained.inputs['spacing'])
        links.new(val.outputs[0], coarse_grained2.inputs['spacing'])
        links.new(val.outputs[0], grid_shader.inputs['spacing'])

        div = nodes.new(type='ShaderNodeMath')
        div.location = (outer_left - 400, 1400)
        div.hide = True
        div.operation = 'DIVIDE'
        div.inputs[1].default_value = 10
        links.new(val.outputs[0], div.inputs[0])
        links.new(div.outputs[0], grid_shader.inputs['thickness'])

        comb = nodes.new(type='ShaderNodeCombineXYZ')
        comb.location = (outer_left - 200, 400)
        links.new(coarse_grained.outputs['Value'], comb.inputs['X'])
        links.new(coarse_grained2.outputs['Value'], comb.inputs['Y'])
        links.new(sep.outputs['Z'], comb.inputs['Z'])

        ramp = nodes.new('ShaderNodeValToRGB')
        ramp.location = (outer_left + 200, 1600)
        ramp.color_ramp.elements[0].color = get_color_from_string('text')
        ramp.color_ramp.elements[1].color = [0, 0, 0, 1]
        links.new(grid_shader.outputs[0], ramp.inputs['Fac'])

        vec_multi = nodes.new('ShaderNodeVectorMath')
        vec_multi.location = (0, 1000)
        vec_multi.operation = 'MULTIPLY'
        links.new(ramp.outputs['Color'], vec_multi.inputs[0])

    for i in range(iterations):
        # create iteration group
        iteration_group = nodes.new(type='ShaderNodeGroup')
        iteration_group.location = (-1200, -200 * i)
        iteration_group.node_tree = iteration_node
        iteration_group.hide = True

        init_val = nodes.new(type='ShaderNodeVectorMath')
        init_val.location = (-1200, 500)
        init_val.hide = True
        links.new(init_val.outputs['Vector'], iteration_group.inputs['z'])
        if coarse_grained:
            links.new(uv.outputs['Object'], sep.inputs['Vector'])
            links.new(comb.outputs['Vector'], iteration_group.inputs['c'])
        else:
            links.new(uv.outputs['Object'], iteration_group.inputs['c'])

        if old_iteration is not None:
            links.new(old_iteration.outputs['z'], iteration_group.inputs['z'])
            links.new(old_iteration.outputs['n'], iteration_group.inputs['n'])

        if i > 0 and interpolation:
            mixer = nodes.new(type='ShaderNodeGroup')
            mixer.node_tree = make_vector_interpolation('VectorInterpol')
            mixer.name = 'VectorInterpol'
            mixer.location = (-600, -200 * i)
            mixer.hide = True
            mixer.inputs['Factor'].default_value = 0

        if interpolation:
            if old_mixer:
                links.new(old_mixer.outputs['Vector'], mixer.inputs['Initial'])
                links.new(old_mixer.outputs['n_out'], mixer.inputs['n_initial'])
            links.new(iteration_group.outputs['z'], mixer.inputs['Final'])
            links.new(iteration_group.outputs['n'], mixer.inputs['n_final'])
            old_mixer = mixer

        old_iteration = iteration_group

    # create phase2color group
    p2c_group = nodes.new(type='ShaderNodeGroup')
    p2c_group.location = (-300, 300)
    p2c_group.node_tree = make_phase2color_group(threshold=threshold)
    p2c_group.hide = True

    if interpolation:
        links.new(mixer.outputs['Vector'], p2c_group.inputs['Complex'])
    else:
        links.new(old_iteration.outputs['z'], p2c_group.inputs['Complex'])

    if phase:
        links.new(p2c_group.outputs['Color'], bsdf.inputs['Base Color'])

    # displacement
    displacement = nodes.new(type='ShaderNodeDisplacement')
    displacement.location = (200, -800)
    displacement.inputs['Midlevel'].default_value = 0
    links.new(displacement.outputs['Displacement'], out.inputs['Displacement'])

    sep_xyz = nodes.new(type='ShaderNodeSeparateXYZ')
    sep_xyz.location = (-300, -800)

    # rescale the heights between zero and one
    scale = nodes.new(type='ShaderNodeMath')
    scale.operation = 'DIVIDE'
    scale.location = (-100, -800)
    links.new(scale.inputs[1], old_iteration.outputs['n'])
    links.new(sep_xyz.outputs['Z'], scale.inputs[0])

    if not phase:
        hue = nodes.new(type='ShaderNodeHueSaturation')
        hue.location = (-200, 200)
        hue.inputs['Color'].default_value = [1, 0, 0, 1]
        links.new(scale.outputs['Value'], hue.inputs['Hue'])
        links.new(hue.outputs['Color'], bsdf.inputs['Base Color'])

    links.new(scale.outputs['Value'], displacement.inputs['Height'])

    if interpolation:
        links.new(mixer.outputs['Vector'], sep_xyz.inputs['Vector'])
    else:
        links.new(old_iteration.outputs['z'], sep_xyz.inputs['Vector'])

    # create RGB mixer for cutoff display
    rgb_mixer = nodes.new(type='ShaderNodeMixRGB')
    rgb_mixer.location = (-150, 300)
    rgb_mixer.hide = True

    if phase:
        links.new(p2c_group.outputs['Color'], rgb_mixer.inputs[1])
    else:
        links.new(hue.outputs['Color'], rgb_mixer.inputs[1])

    links.new(scale.outputs['Value'], rgb_mixer.inputs[2])  # create increasingly white plane color
    if coarse_grained:
        links.new(rgb_mixer.outputs[0], vec_multi.inputs[1])
        links.new(vec_multi.outputs[0], bsdf.inputs['Base Color'])
    else:
        links.new(rgb_mixer.outputs[0], bsdf.inputs['Base Color'])

    less = nodes.new(type='ShaderNodeMath')
    less.location = (-300, 100)
    less.operation = 'LESS_THAN'
    less.inputs[1].default_value = 0
    less.hide = True

    links.new(sep_xyz.outputs['Z'], less.inputs[0])
    if old_mixer is not None:
        links.new(old_mixer.outputs['n_out'], less.inputs[1])
    else:
        less.inputs[1].default_value = iterations
    links.new(less.outputs['Value'], rgb_mixer.inputs[0])

    return material


def make_hue_material(**kwargs):
    if 'hue_value' in kwargs:
        value = kwargs.pop('hue_value')
    else:
        value = 0

    material = bpy.data.materials.new(name="hue_material")
    # for eevee
    material.use_screen_refraction = True
    # for cycles
    material.use_nodes = True
    tree = material.node_tree
    nodes = material.node_tree.nodes
    links = tree.links
    bsdf = nodes.get("Principled BSDF")

    hue = nodes.new(type='ShaderNodeHueSaturation')
    hue.location = (-400, 400)
    hue.label = 'hue_node'

    hue.inputs['Hue'].default_value = value
    hue.inputs['Saturation'].default_value = 1
    hue.inputs['Color'].default_value = [0, 1, 1, 1]
    links.new(hue.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(hue.outputs['Color'], bsdf.inputs[EMISSION])

    return material


def make_iteration_material(**kwargs):
    if 'iterations' in kwargs:
        iterations = kwargs.pop('iterations')

    if 'threshold' in kwargs:
        threshold = kwargs.pop('threshold')
    else:
        threshold = 2

    if 'ramp_colors' in kwargs:
        ramp_colors = kwargs.pop('ramp_colors')
    else:
        ramp_colors = ["background", "text"]

    material = bpy.data.materials.new(name="mandelbrot")
    # for eevee
    material.use_screen_refraction = True
    # for cycles
    material.cycles.displacement_method = 'DISPLACEMENT'
    material.use_nodes = True
    tree = material.node_tree
    nodes = material.node_tree.nodes
    links = tree.links
    bsdf = nodes.get("Principled BSDF")
    out = nodes.get("Material Output")

    outer_left = -1200

    uv = nodes.new(type='ShaderNodeTexCoord')
    uv.location = (outer_left - 400, 400)

    uv_coords = uv.outputs['Object']

    # discretizer
    discretizer = create_group_from_vector_function(nodes,
                                                    ["x,spacing_x,/,floor,spacing_x,*",
                                                     "y,spacing_y,/,floor,spacing_y,*", "z"],
                                                    parameters=["spacing"], name="Discretizer")

    discretizer.location = (outer_left - 200, 600)
    links.new(uv_coords, discretizer.inputs[0])
    coords = discretizer.outputs[0]

    spacing_vec = nodes.new(type='ShaderNodeCombineXYZ')
    spacing_vec.location = (outer_left - 400, 1000)

    val = nodes.new(type='ShaderNodeValue')
    val.location = (outer_left - 600, 1000)
    val.label = 'Spacing'
    val.outputs[0].default_value = 0.1

    thickness_vec = nodes.new(type='ShaderNodeCombineXYZ')
    thickness_vec.location = (outer_left - 400, 1200)

    thickness = nodes.new(type='ShaderNodeValue')
    thickness.location = (outer_left - 600, 1200)
    thickness.label = 'Thickness'
    thickness.outputs[0].default_value = 0.03
    links.new(val.outputs[0], spacing_vec.inputs[0])
    links.new(val.outputs[0], spacing_vec.inputs[1])
    links.new(val.outputs[0], spacing_vec.inputs[2])

    links.new(thickness.outputs[0], thickness_vec.inputs[0])
    links.new(thickness.outputs[0], thickness_vec.inputs[1])
    links.new(thickness.outputs[0], thickness_vec.inputs[2])

    links.new(spacing_vec.outputs[0], discretizer.inputs[1])

    # grid
    grid = create_group_from_vector_function(nodes, ["x,spacing_x,/,floor,spacing_x,*,x,-,abs,thickness_x,<",
                                                     "y,spacing_y,/,floor,spacing_y,*,y,-,abs,thickness_y,<",
                                                     "z"
                                                     ], parameters=["spacing", "thickness"], name="GridNode")

    grid.location = (outer_left + 200, 1200)
    links.new(uv_coords, grid.inputs[0])
    links.new(spacing_vec.outputs[0], grid.inputs[1])
    links.new(thickness_vec.outputs[0], grid.inputs[2])

    # create an iterator_group for each corner of the pixel
    iterator_group_dl = create_iterator_group(nodes, ["x,x,*,y,y,*,-,c_x,+", "2,x,y,*,*,c_y,+"], ["c"], iterations,
                                              name="iteratorGroupDL" + str(iterations))
    iterator_group_dl.location = (outer_left + 300, 400)
    iterator_group_dr = create_iterator_group(nodes, ["x,x,*,y,y,*,-,c_x,+", "2,x,y,*,*,c_y,+"], ["c"], iterations,
                                              name="iteratorGroupDR" + str(iterations))
    iterator_group_dr.location = (outer_left + 300, 200)
    iterator_group_ul = create_iterator_group(nodes, ["x,x,*,y,y,*,-,c_x,+", "2,x,y,*,*,c_y,+"], ["c"], iterations,
                                              name="iteratorGroupUL" + str(iterations))
    iterator_group_ul.location = (outer_left + 300, 0)
    iterator_group_ur = create_iterator_group(nodes, ["x,x,*,y,y,*,-,c_x,+", "2,x,y,*,*,c_y,+"], ["c"], iterations,
                                              name="iteratorGroupUR" + str(iterations))
    iterator_group_ur.location = (outer_left + 300, -200)

    # create pixel corners
    coords_dl = coords
    coords_dr = create_group_from_vector_function(nodes, ['x,spacing_x,+', 'y,0,+'], ['spacing'], name="groupDR")
    coords_dr.location = (outer_left, 400)
    coords_ul = create_group_from_vector_function(nodes, ['x,0,+', 'y,spacing_y,+'], ['spacing'], name="groupUL")
    coords_ul.location = (outer_left, 200)
    coords_ur = create_group_from_vector_function(nodes, ['x,spacing_x,+', 'y,spacing_y,+'], ['spacing'],
                                                  name="groupUR")
    coords_ur.location = (outer_left, 0)

    links.new(spacing_vec.outputs[0], coords_dr.inputs[1])
    links.new(spacing_vec.outputs[0], coords_ul.inputs[1])
    links.new(spacing_vec.outputs[0], coords_ur.inputs[1])
    links.new(discretizer.outputs[0], coords_dr.inputs[0])
    links.new(discretizer.outputs[0], coords_ul.inputs[0])
    links.new(discretizer.outputs[0], coords_ur.inputs[0])

    links.new(discretizer.outputs[0], iterator_group_dl.inputs[0])
    links.new(coords_dr.outputs[0], iterator_group_dr.inputs[0])
    links.new(coords_ul.outputs[0], iterator_group_ul.inputs[0])
    links.new(coords_ur.outputs[0], iterator_group_ur.inputs[0])

    # check, whether all for corners of the pixel stay below 2 after iterations
    and_group = create_group_from_vector_function(nodes,
                                                  ['v,length,2,<,dr,length,2,<,*,ul,length,2,<,*,ur,length,2,<,*'],
                                                  parameters=['dr', 'ul', 'ur'], name='andGroup')
    and_group.location = (outer_left + 500, 400)
    links.new(iterator_group_dl.outputs[0], and_group.inputs[0])
    links.new(iterator_group_dr.outputs[0], and_group.inputs[1])
    links.new(iterator_group_ul.outputs[0], and_group.inputs[2])
    links.new(iterator_group_ur.outputs[0], and_group.inputs[3])

    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.location = (outer_left + 900, 400)
    ramp.color_ramp.elements[0].color = get_color_from_string(ramp_colors[0])
    ramp.color_ramp.elements[1].color = get_color_from_string(ramp_colors[1])

    ramp2 = nodes.new('ShaderNodeValToRGB')
    ramp2.location = (outer_left + 900, 800)
    ramp2.color_ramp.elements[1].color = [0, 0, 0, 1]
    ramp2.color_ramp.elements[1].position = 0.27
    ramp2.color_ramp.elements[0].color = [1, 1, 1, 1]
    links.new(grid.outputs[0], ramp2.inputs['Fac'])

    vec_multi = nodes.new('ShaderNodeVectorMath')
    vec_multi.location = (outer_left + 1100, 600)
    vec_multi.operation = 'MULTIPLY'
    links.new(ramp.outputs['Color'], vec_multi.inputs[0])
    links.new(ramp2.outputs['Color'], vec_multi.inputs[1])

    # comparators for pixel-wise appearance

    pixel_iterator = create_group_from_vector_function(nodes, parameters=["position", "spacing"],
                                                       functions=[
                                                           "position_x,x,<,position_x,spacing_x,+,x,>,*,y,position_y,<,*,x,position_x,<,+,1,min"],
                                                       name='pixelIterator')
    pixel_iterator.location = (0, 1000)
    links.new(discretizer.outputs[0], pixel_iterator.inputs[0])
    links.new(spacing_vec.outputs[0], pixel_iterator.inputs[2])

    val_x = nodes.new(type='ShaderNodeValue')
    val_x.location = (-400, 1100)
    val_x.name = 'XCoordParam'
    val_x.label = 'XCoordParam'
    val_x.outputs[0].default_value = 0

    val_y = nodes.new(type='ShaderNodeValue')
    val_y.location = (-400, 900)
    val_y.name = 'YCoordParam'
    val_y.label = 'YCoordParam'
    val_y.outputs[0].default_value = 0

    combine = nodes.new('ShaderNodeCombineXYZ')
    combine.location = (-200, 1000)
    links.new(val_x.outputs[0], combine.inputs[0])
    links.new(val_y.outputs[0], combine.inputs[1])
    links.new(combine.outputs[0], pixel_iterator.inputs[1])
    links.new(and_group.outputs[0], ramp.inputs[0])

    mixer = nodes.new(type='ShaderNodeMixRGB')
    mixer.location = (-100, 0)
    mixer.inputs[1].default_value = get_color_from_string("gray_2")
    links.new(pixel_iterator.outputs[0], mixer.inputs[0])
    links.new(vec_multi.outputs[0], mixer.inputs[2])

    links.new(mixer.outputs[0], bsdf.inputs['Base Color'])
    links.new(mixer.outputs[0], bsdf.inputs[EMISSION])

    # displacement

    mul = nodes.new('ShaderNodeMath')
    mul.location = (-400, 0)
    mul.operation = 'MULTIPLY'
    mul.inputs[1].default_value = 0.5
    links.new(pixel_iterator.outputs[0], mul.inputs[0])
    links.new(and_group.outputs[0], mul.inputs[1])

    displacement = nodes.new(type='ShaderNodeDisplacement')
    displacement.location = (0, -800)
    displacement.inputs['Midlevel'].default_value = 0
    displacement.inputs['Scale'].default_value = 0.0225
    links.new(mul.outputs[0], displacement.inputs['Height'])
    links.new(displacement.outputs['Displacement'], out.inputs['Displacement'])

    return material


def get_material_of(bob):
    obj = get_obj(bob)
    return obj.data.materials[0]


def get_nodes_of_material(bob, name_part='Mix'):
    obj = get_obj(bob)
    material = get_material_of(bob)
    tree = material.node_tree
    nodes = material.node_tree.nodes

    return [node for node in nodes if name_part in node.name]


def change_mixer(mixer, begin_frame=0, transition_frames=DEFAULT_ANIMATION_TIME * FRAME_RATE,
                 begin_time=None, transition_time=None):
    if begin_time:
        begin_frame = begin_time * FRAME_RATE
    else:
        begin_time = begin_frame / FRAME_RATE
    if transition_time:
        transition_frames = transition_time * FRAME_RATE
    else:
        transition_time = transition_frames / FRAME_RATE

    mixer.default_value = 0
    insert_keyframe(mixer, "default_value", begin_frame)
    mixer.default_value = 1
    insert_keyframe(mixer, "default_value", begin_frame + transition_frames)

    return begin_time + transition_time


def change_default_value(slot, from_value, to_value, begin_time=None, transition_time=None, data_path="default_value",
                         begin_frame=0, transition_frames=DEFAULT_ANIMATION_TIME * FRAME_RATE):
    if begin_time:
        begin_frame = begin_time * FRAME_RATE
    else:
        begin_time = begin_frame / FRAME_RATE
    if transition_time is not None:
        transition_frames = np.maximum(1, transition_time * FRAME_RATE)
    else:
        transition_time = transition_frames / FRAME_RATE
    if from_value is not None:
        slot.default_value = from_value
        insert_keyframe(slot, data_path, begin_frame)
    slot.default_value = to_value
    insert_keyframe(slot, data_path, begin_frame + transition_frames)

    return begin_time + transition_time


def change_default_boolean(slot, from_value, to_value, begin_time=None, transition_time=None, data_path="boolean",
                           begin_frame=0):
    if begin_time:
        begin_frame = begin_time * FRAME_RATE
    else:
        begin_time = begin_frame / FRAME_RATE
    if from_value is not None:
        slot.boolean = from_value
        insert_keyframe(slot, data_path, begin_frame)
    slot.boolean = to_value
    insert_keyframe(slot, data_path, begin_frame + 1)

    return (begin_frame + 1) / FRAME_RATE


def change_default_vector(slot, from_value, to_value, begin_time=None, transition_time=None, data_path="vector",
                          begin_frame=0, transition_frames=DEFAULT_ANIMATION_TIME * FRAME_RATE):
    if begin_time:
        begin_frame = begin_time * FRAME_RATE
    else:
        begin_time = begin_frame / FRAME_RATE
    if transition_time:
        transition_frames = transition_time * FRAME_RATE
    else:
        transition_time = transition_frames / FRAME_RATE
    if from_value is not None:
        slot.vector = from_value
        insert_keyframe(slot, data_path, begin_frame)
    slot.vector = to_value
    insert_keyframe(slot, data_path, begin_frame + transition_frames)

    return begin_time + transition_time


def change_value(value, from_value, to_value, begin_time=None, transition_time=None, data_path="default_value",
                 begin_frame=0, transition_frames=DEFAULT_ANIMATION_TIME * FRAME_RATE):
    if begin_time:
        begin_frame = begin_time * FRAME_RATE
    else:
        begin_time = begin_frame / FRAME_RATE
    if transition_time:
        transition_frames = transition_time * FRAME_RATE
    else:
        transition_time = transition_frames / FRAME_RATE
    if from_value is not None:
        value.outputs[0].default_value = from_value
        insert_keyframe(value.outputs[0], data_path, begin_frame)
    value.outputs[0].default_value = to_value
    insert_keyframe(value.outputs[0], data_path, begin_frame + transition_frames)

    return begin_time + transition_time


def make_image__stretched_over_geometry(src, v_min, v_max):
    color = bpy.data.materials.new(name='image_' + src)
    color.use_nodes = True
    nodes = color.node_tree.nodes
    bsdf = nodes['Principled BSDF']
    img = nodes.new('ShaderNodeTexImage')
    img.location = (-300, 0)
    if src:
        img.image = bpy.data.images.load(os.path.join(IMG_DIR, src))
        img.extension = 'EXTEND'
    color.node_tree.links.new(img.outputs['Color'], bsdf.inputs['Base Color'])
    color.node_tree.links.new(img.outputs['Color'], bsdf.inputs[EMISSION])
    # color.node_tree.links.new(img.outputs['Alpha'], bsdf.inputs['Alpha']) # this disables the fade out of objects

    # stretch image over the entire object
    combine = nodes.new('ShaderNodeCombineXYZ')
    combine.location = (-500, 0)
    color.node_tree.links.new(combine.outputs['Vector'], img.inputs['Vector'])

    map_x = nodes.new('ShaderNodeMapRange')
    map_x.location = (-700, 400)
    map_x.inputs['From Min'].default_value = v_min[0]
    map_x.inputs['From Max'].default_value = v_max[0]
    color.node_tree.links.new(map_x.outputs['Result'], combine.inputs['X'])

    map_y = nodes.new('ShaderNodeMapRange')
    map_y.location = (-700, -400)
    map_y.inputs['From Min'].default_value = v_min[1]
    map_y.inputs['From Max'].default_value = v_max[1]
    color.node_tree.links.new(map_y.outputs['Result'], combine.inputs['Y'])

    sep = nodes.new('ShaderNodeSeparateXYZ')
    sep.location = (-900, 0)
    color.node_tree.links.new(sep.outputs['X'], map_x.inputs['Value'])
    color.node_tree.links.new(sep.outputs['Y'], map_y.inputs['Value'])

    geometry = nodes.new('ShaderNodeNewGeometry')
    geometry.location = (-1100, 0)
    color.node_tree.links.new(geometry.outputs['Position'], sep.inputs['Vector'])

    return color


def make_gradient_material(**kwargs):
    colors = get_from_kwargs(kwargs, 'colors', ['gray_1,gray_9'])
    material = bpy.data.materials.new(name='gradient_' + colors[0] + "_" + colors[1])
    ramp_positions = get_from_kwargs(kwargs, 'ramp_positions', [0, 1])
    coordinate_type = get_from_kwargs(kwargs, 'coordinate_type', 'UV')
    coordinate = get_from_kwargs(kwargs, 'coordinate', 'Y')

    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    bsdf = nodes['Principled BSDF']
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.location = (-400, 0)
    for i, pos in enumerate(ramp_positions):
        ramp.color_ramp.elements[i].position = pos
    for i, color in enumerate(colors):
        ramp.color_ramp.elements[i].color = get_color_from_string(color)

    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(ramp.outputs['Color'], bsdf.inputs[EMISSION])

    separate_xyz = nodes.new('ShaderNodeSeparateXYZ')
    separate_xyz.location = (-600, 0)
    links.new(separate_xyz.outputs[coordinate], ramp.inputs['Fac'])

    coords = nodes.new(type='ShaderNodeTexCoord')
    coords.location = (-900, 0)
    links.new(coords.outputs[coordinate_type], separate_xyz.inputs['Vector'])

    return material


def make_dashed_material(**kwargs):
    colors = get_from_kwargs(kwargs, 'colors', ['drawing'])
    material = bpy.data.materials.new(name='dashed_' + colors[0])
    coordinate = get_from_kwargs(kwargs, 'coordinate', 'Z')
    scale = get_from_kwargs(kwargs, 'dash_scale', 2)
    phase_offset = get_from_kwargs(kwargs, 'phase_offset', 0)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    bsdf = nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = get_color_from_string(colors[0])

    greater = nodes.new('ShaderNodeMath')
    greater.location = (-400, 0)
    greater.operation = 'GREATER_THAN'
    greater.inputs[1].default_value = 0.5
    links.new(greater.outputs['Value'], bsdf.inputs['Alpha'])

    wave = nodes.new('ShaderNodeTexWave')
    wave.location = (-600, 0)
    wave.bands_direction = coordinate
    wave.inputs['Scale'].default_value = scale
    wave.inputs['Phase Offset'].default_value = phase_offset
    links.new(wave.outputs['Color'], greater.inputs['Value'])

    return material


def make_image_material(src=None, **kwargs):
    if 'name' in kwargs:
        name = kwargs.pop('name')
    else:
        name = 'image_' + src
    color = bpy.data.materials.new(name=name)
    color.use_nodes = True
    nodes = color.node_tree.nodes
    bsdf = nodes['Principled BSDF']
    img = nodes.new('ShaderNodeTexImage')
    img.location = (-400, 0)
    coords = nodes.new('ShaderNodeTexCoord')
    coords.location = (-1000, 0)
    map = nodes.new('ShaderNodeMapping')
    map.location = (-600, 0)
    location = get_from_kwargs(kwargs, 'location', [0, 0, 0])
    scale = get_from_kwargs(kwargs, 'scale', [1, 1, 1])
    rotation = get_from_kwargs(kwargs, 'rotation', [0, 0, 0])
    extension = get_from_kwargs(kwargs, 'extension', 'EXTEND')
    coordinates = get_from_kwargs(kwargs, 'coordinates', 'Generated')

    map.inputs[1].default_value = location
    map.inputs[2].default_value = rotation
    map.inputs[3].default_value = scale
    if src:
        img.image = bpy.data.images.load(os.path.join(IMG_DIR, src))
        img.extension = extension
    links = color.node_tree.links
    links.new(img.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(img.outputs['Color'], bsdf.inputs[EMISSION])

    alpha_factor = nodes.new(type='ShaderNodeMath')
    alpha_factor.name = 'AlphaFactor'
    alpha_factor.label = 'AlphaFactor'
    alpha_factor.operation = 'MULTIPLY'
    alpha_factor.location = (-200, 0)
    alpha_factor.inputs[0].default_value = 1

    links.new(img.outputs['Alpha'], alpha_factor.inputs[1])
    links.new(alpha_factor.outputs[0], bsdf.inputs['Alpha'])
    links.new(map.outputs['Vector'], img.inputs['Vector'])
    links.new(coords.outputs[coordinates], map.inputs['Vector'])

    emission = get_from_kwargs(kwargs, 'emission', 0)
    bsdf.inputs['Emission Strength'].default_value = emission

    return color


def set_movie_to_material(bob, src=None, duration=0, **kwargs):
    obj = get_obj(bob)
    mat = obj.material_slots[0].material

    nodes = mat.node_tree.nodes
    img = nodes['Image Texture']
    if src:
        img.image = bpy.data.images.load(os.path.join(VID_DIR, src))
        img.image_user.frame_duration = int(duration * FRAME_RATE)
        img.image_user.use_auto_refresh = True
        offset = get_from_kwargs(kwargs, 'offset', 0)
        img.image_user.frame_offset = offset
    inverted = get_from_kwargs(kwargs, 'inverted', False)
    if inverted:
        links = mat.node_tree.links
        invert = nodes.new(type='ShaderNodeInvert')
        invert.location = (-200, 500)

        # find the link that has to be replaced
        for l in links:
            if l.from_node == img:
                special_link = l
                from_socket = special_link.from_socket
                to_socket = special_link.to_socket
                print(special_link, from_socket, to_socket)
                break

        if special_link:
            # link invert node into the previously existing link
            links.new(from_socket, invert.inputs['Color'])
            links.new(invert.outputs['Color'], to_socket)


def set_movie_start(bob, begin_frame=0):
    obj = get_obj(bob)
    mat = obj.material_slots[0].material

    nodes = mat.node_tree.nodes
    img = nodes['Image Texture']
    if img:
        img.image_user.frame_start = int(begin_frame)


def set_material(bob, material, slot=0):
    obj = bob.ref_obj
    if isinstance(material, str):
        material = bpy.data.materials.get(material)

    if material:
        if len(obj.data.materials) == 0:
            obj.data.materials.append(material)
        else:
            if len(obj.material_slots) <= slot:
                obj.data.materials.append(material)
            else:
                obj.material_slots[slot].material = material
    else:
        print("No material found with name ", material)


def asign_material_to_faces(bob, material_index, normal=Vector([0, 0, 1])):
    """
    asign the material to all faces with a given normal
    :param bob:
    :param material_index:
    :param normal:
    :return:
    """

    obj = get_obj(bob)
    if len(obj.material_slots) <= material_index:
        pass
    else:
        sel_faces = select_faces(obj.data.polygons, normal)
        for face in sel_faces:
            face.material_index = material_index


def select_faces(faces, normal):
    sel_faces = []
    for face in faces:
        if np.abs(face.normal.dot(normal) - 1) < 0.1:
            sel_faces.append(face)
    return sel_faces


def stretch_uv_map_to_faces(bob, normal=Vector([0, 0, 1])):
    """
    stretch the uv map to all the selected faces uniformly 
    :param bob:
    :param normal:
    :return:
    """
    obj = get_obj(bob)
    bm = bmesh.new()
    bm.from_mesh(obj.data)  # Fills it in using the cylinder
    uv_layer = bm.loops.layers.uv.active

    sel_faces = select_faces(bm.faces, normal)
    min_x = min_y = np.inf
    max_x = max_y = -np.inf
    for face in sel_faces:
        for loop in face.loops:
            vert = loop.vert.co
            if vert.x < min_x:
                min_x = vert.x
            if vert.x > max_x:
                max_x = vert.x
            if vert.y < min_y:
                min_y = vert.y
            if vert.y > max_y:
                max_y = vert.y

    for face in sel_faces:
        for loop in face.loops:
            # uv = loop[uv_layer].uv
            # print("Loop UV: %f, %f" % uv[:])

            vert = loop.vert.co
            u = lin_map(vert.x, min_x, max_x, 0, 1)
            v = lin_map(vert.y, min_y, max_y, 0, 1)

            loop[uv_layer].uv = [u, v]

            uv = loop[uv_layer].uv

            # print("Loop UV: %f, %f" % uv[:])
            # print("Loop Vert: (%f, %f, %f)" % vert[:])

    bm.to_mesh(obj.data)
    bm.free()


def separate(bob, type="SELECTED"):
    obj = get_obj(bob)
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.separate(type=type)
    bpy.ops.object.editmode_toggle()


def setup_material_for_alpha_masking(bob):
    """
    This function prepares the material to allow for image textures that use the alpha-channel for masking
    It is used for varying numbers displayed on flags

    :param bob:
    :return:
    """

    obj = bob.ref_obj
    if obj.data and obj.data.materials:
        material = obj.data.materials[0]

    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    bsdf = nodes['Principled BSDF']
    out = nodes['Material Output']

    # preserve old bsdf with color as one entry for the shader mixer
    shader_mixer = nodes.new(type='ShaderNodeMixShader')
    links.new(shader_mixer.outputs[0], out.inputs[0])
    links.new(bsdf.outputs['BSDF'], shader_mixer.inputs[1])

    # create a new bsdf that receives the alpha mask as color and emission imput
    bsdf2 = nodes.new(type='ShaderNodeBsdfPrincipled')
    links.new(bsdf2.outputs['BSDF'], shader_mixer.inputs[2])
    bsdf2.inputs['Emission Strength'].default_value = 1


def add_image_texture(bob, image, begin_frame=None, frame_duration=DEFAULT_ANIMATION_TIME * FRAME_RATE, begin_time=None,
                      transition_time=DEFAULT_ANIMATION_TIME):
    obj = bob.ref_obj
    if begin_frame:
        start = begin_frame
    elif begin_time:
        start = begin_time * FRAME_RATE
    else:
        start = 0

    if frame_duration:
        dur = frame_duration
    elif transition_time:
        dur = transition_time * FRAME_RATE
    else:
        dur = DEFAULT_ANIMATION_TIME * FRAME_RATE

    if obj.data and obj.data.materials:
        material = obj.data.materials[0]
        dialer = create_image_mixing_find_previous_mixer(material, image)
        if dialer:
            dialer.default_value = 0
            insert_keyframe(dialer, 'default_value', int(start))
            dialer.default_value = 1
            insert_keyframe(dialer, 'default_value', int(start) + np.maximum(1, dur))


def create_image_mixing_find_previous_mixer(material, image_path):
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # find last mixer assuming that they are ordered 'Mix', 'Mix.001', 'Mix.002', ...
    last_mixer = None
    for node in nodes:
        if 'Mix' == node.name:
            last_mixer = node

    count = 0
    if last_mixer:
        next_mixer = True
        count = 1
        while next_mixer:
            next_mixer = False
            for node in nodes:
                if 'Mix' in node.name and str(count) in node.name:
                    last_mixer = node
                    next_mixer = True
                    count += 1

    if 'Image Texture' in nodes:
        last_image = nodes['Image Texture']
    else:
        last_image = None

    bsdf = nodes['Principled BSDF.001']  # take the second BSDF, the first sets the base color

    img = nodes.new(type='ShaderNodeTexImage')
    img.location = (-300 * (count + 1), 0)
    img.image = bpy.data.images.load(image_path)
    img.extension = 'EXTEND'

    mixer = nodes.new(type='ShaderNodeMixRGB')
    mixer.location = (-300 * count, -200)
    mixer.inputs['Fac'].default_value = 0

    if not last_image:
        # just attach image with a mixer to the second bsdf
        links.new(mixer.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(mixer.outputs['Color'], bsdf.inputs[EMISSION])
        links.new(img.outputs['Alpha'], mixer.inputs[1])
    else:
        links.new(img.outputs['Alpha'], mixer.inputs[1])
        links.new(mixer.outputs['Color'], last_mixer.inputs[2])
        return last_mixer.inputs['Fac']


def change_color(bob, new_color, begin_frame, final_frame):
    """
    Change color by adding a new color mixer
    material properties are not affected
    :param bob:
    :param new_color:
    :param begin_frame:
    :param final_frame:
    :return:
    """
    obj = get_obj(bob)
    if obj.data and obj.data.materials:
        material = obj.data.materials[0]
        dialer = create_color_mixing_find_previous_color(material, new_color)
        dialer.default_value = 0
        insert_keyframe(dialer, 'default_value', begin_frame)
        dialer.default_value = 1
        if int(final_frame) == int(begin_frame):
            final_frame = begin_frame + 1
        insert_keyframe(dialer, 'default_value', final_frame)
    else:
        for child in obj.children:
            change_color(child, new_color, begin_frame, final_frame)


def create_color_mixing_find_previous_color(material, color):
    return create_color_mixing(material, None, color)


def create_color_mixing(material, color1, color2):
    """
    create a mixing between two colors
    :param material:
    :param color1: string
    :param color2: string
    :return: dialer for the mixer
    """

    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # find last mixer assuming that they are ordered 'Mix', 'Mix.001', 'Mix.002', ...
    last_mixer = None
    for node in nodes:
        if 'Mix' == node.name:
            last_mixer = node

    if last_mixer:
        next_mixer = True
        count = 1
        while next_mixer:
            next_mixer = False
            for node in nodes:
                if 'Mix' in node.name and str(count) in node.name:
                    last_mixer = node
                    next_mixer = True
                    count += 1

    bsdf = nodes['Principled BSDF']
    mixer = nodes.new(type='ShaderNodeMixRGB')
    mixer.name = 'Mix'  # since 4.0 the name has changed to "Mix (Legacy)", therefore enforce old name scheme
    mixer.label = 'Mix'
    mixer.inputs['Fac'].default_value = 0

    # get previous color
    if not color1 and not last_mixer:
        mixer.inputs['Color1'].default_value = bsdf.inputs['Base Color'].default_value
    elif not color1:
        mixer.inputs['Color1'].default_value = last_mixer.inputs['Color2'].default_value
    else:
        mixer.inputs['Color1'].default_value = get_color_from_string(color1)
    mixer.inputs['Color2'].default_value = get_color_from_string(color2)

    # link new mixer to the existing structure
    if not last_mixer:
        links.new(mixer.outputs[0], bsdf.inputs['Base Color'])
        links.new(mixer.outputs[0], bsdf.inputs[EMISSION])
    else:
        links.new(mixer.outputs[0], last_mixer.inputs['Color2'])
    return mixer.inputs['Fac']


def create_shader_from_script(material, osl_script, emission_strength=0.3):
    """
       this is an unskillful way to color bezier curves with position-dependent phase colors

       :param osl_script:
       :param material:
       :param emission_strength:
       :return:
       """

    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    node_bsdf = nodes['Principled BSDF']
    node_bsdf.inputs['Emission Strength'].default_value = emission_strength

    # setup CPU rendering and enable open-shader language
    # only use this for a limited number of frames

    if bpy.data.scenes["Scene"].cycles.device != 'CPU':
        bpy.data.scenes["Scene"].cycles.device = 'CPU'
    if not bpy.data.scenes["Scene"].cycles.shading_system:
        bpy.data.scenes["Scene"].cycles.shading_system = True

    dist = 3

    # prepare the calculation of the phase from the imaginary and real part of the resulting complex function
    node_geometry = nodes.new(type='ShaderNodeNewGeometry')
    node_geometry.location = (-dist * 200, 0)

    dist -= 1

    node_script = nodes.new(type='ShaderNodeScript')
    node_script.location = (-dist * 200, 0)
    node_script.mode = 'EXTERNAL'
    node_script.filepath = OSL_DIR + "/" + osl_script + '.osl'

    links.new(node_geometry.outputs['Position'], node_script.inputs['vec'])

    dist -= 1
    node_hue = nodes.new(type='ShaderNodeHueSaturation')
    node_hue.location = (-dist * 200, 0)

    node_hue.inputs['Color'].default_value = [1, 0, 0, 1]
    links.new(node_script.outputs['phi'], node_hue.inputs['Hue'])

    links.new(node_hue.outputs['Color'], node_bsdf.inputs['Base Color'])
    links.new(node_hue.outputs['Color'], node_bsdf.inputs[EMISSION])


def create_voronoi_shader(material, colors, scale=[1, 1, 1], emission_strength=0.3):
    """
    this is an unskillful way to parse the function, provided in inverse polish notation and create the corresponding
    math nodes for the texture in blender

    :param material:
    :param hue_functions:
    :param scale:
    :param emission_strength:
    :return:
    """
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    node_bsdf = nodes['Principled BSDF']
    node_bsdf.inputs['Emission Strength'].default_value = emission_strength

    number_of_shadings = len(colors)

    node_voronoi = nodes.new(type='ShaderNodeTexVoronoi')
    node_voronoi.location = (-number_of_shadings * 200, 0)
    node_voronoi.inputs['Scale'].default_value = 20

    out_voronoi = node_voronoi.outputs['Color']

    dialer = []
    dialer_nodes = []
    ramps = []

    for s in range(0, number_of_shadings):
        if s < len(colors):
            palette = colors[s]
        else:
            # if there are less color lists than functions, the last palette is used for the rest of the functions
            palette = colors[-1]

        node_ramp = nodes.new(type='ShaderNodeValToRGB')
        node_ramp.location = ((-number_of_shadings + s + 1) * 200, 0)
        node_ramp.color_ramp.elements.remove(node_ramp.color_ramp.elements[1])

        delta = 1 / len(palette)
        for i, p in enumerate(palette):
            if i > 0:
                node_ramp.color_ramp.elements.new(delta * i)
            node_ramp.color_ramp.elements[i].color = tuple(p)

        node_ramp.color_ramp.interpolation = 'CONSTANT'
        links.new(out_voronoi, node_ramp.inputs[0])
        ramps.append(node_ramp)

    # merge all hue_nodes into the bsdf with mixrgb nodes.

    for i, hue in enumerate(ramps):
        if (i + 1) < len(ramps):
            # create a rgb mixer for the next color_layer
            mixer = nodes.new(type='ShaderNodeMixRGB')
            mixer.inputs[0].default_value = 0
            dialer.append(mixer.inputs[0])
            mixer.location = (4 * 200, 400 * i)
            links.new(hue.outputs[0], mixer.inputs[1])
            if len(dialer_nodes) > 0:
                links.new(mixer.outputs['Color'], dialer_nodes[-1].inputs[2])
            dialer_nodes.append(mixer)
        else:
            if len(dialer_nodes) > 0:
                links.new(hue.outputs['Color'], dialer_nodes[-1].inputs[2])

    # link the hue nodes and mixers to the principled bsdf
    if len(dialer_nodes) == 0:
        # only one hue_node
        links.new(ramps[0].outputs['Color'], node_bsdf.inputs['Base Color'])
        links.new(ramps[0].outputs['Color'], node_bsdf.inputs[EMISSION])
    else:
        links.new(dialer_nodes[0].outputs['Color'], node_bsdf.inputs['Base Color'])
        links.new(dialer_nodes[0].outputs['Color'], node_bsdf.inputs[EMISSION])

    node_bsdf.location = (4 * 200 + 100, 0)
    nodes['Material Output'].location = (6 * 200, 0)

    return dialer


def create_color_map_for_mesh(bob, colors, name, **kwargs):
    obj = get_obj(bob)
    color_map_collection = obj.data.vertex_colors
    color_map = color_map_collection.new(name="color_map_" + name)

    i = 0
    for poly in obj.data.polygons:
        for idx in poly.loop_indices:
            loop = obj.data.loops[idx]
            v = loop.vertex_index
            color_map.data[i].color = colors[v]
            i += 1

    print("Complex material created for " + color_map.name)

    phase_color = bpy.data.materials.new(name="Phase Color")
    phase_color.use_nodes = True
    links = phase_color.node_tree.links
    nodes = phase_color.node_tree.nodes
    p_bsdf = nodes.get("Principled BSDF")

    vert_col = nodes.new(type='ShaderNodeVertexColor')
    vert_col.layer_name = color_map.name

    links.new(vert_col.outputs[0], p_bsdf.inputs['Base Color'])
    links.new(vert_col.outputs[0], p_bsdf.inputs[EMISSION])

    # set default values for a nice texture
    alpha = bob.get_from_kwargs('alpha', 0.5)
    metallic = bob.get_from_kwargs('metallic', 1)
    roughness = bob.get_from_kwargs('roughness', 0.6)
    emission_strength = bob.get_from_kwargs('emission_strength', 0.2)
    transmission = bob.get_from_kwargs('transmission', 0)

    set_alpha_for_material(phase_color, alpha)
    set_metallic_for_material(phase_color, metallic)
    set_roughness_for_material(phase_color, roughness)
    set_emission_strength_for_material(phase_color, emission_strength)
    set_transmission_for_material(phase_color, transmission)

    if len(obj.data.materials) == 0:
        obj.data.materials.append(phase_color)
    else:
        obj.material_slots[0].material = phase_color


def set_vertex_colors(bob, colors):
    obj = get_obj(bob)
    vert_list = obj.data.vertices
    color_map_collection = obj.data.vertex_colors
    color_maps = []

    color_map = color_map_collection.new(name="color_map_" + obj.name)
    i = 0
    for poly in obj.data.polygons:
        for idx in poly.loop_indices:
            loop = obj.data.loops[idx]
            v = loop.vertex_index
            if i < len(colors):
                color = colors[i]
            else:
                color = colors[-1]
            color_map.data[i].color = color
            i += 1
    color_maps.append(color_map)
    print("Vertex colors created " + color_map.name)


def create_iterator_group(nodes, functions, parameters, iterations, name='IteratorGroup'):
    """
    creates a group that contains a number of iteration sub-groups
    :param nodes:
    :param functions: iteration function
    :param parameters: parameter for iteration
    :param iterations: number of iterations
    :param name:
    :return:
    """

    tree = bpy.data.node_groups.new(type='ShaderNodeTree', name=name)
    tree_nodes = tree.nodes
    tree_links = tree.links
    delta = 200
    length = -1

    group_inputs = tree_nodes.new('NodeGroupInput')
    group_outputs = tree_nodes.new('NodeGroupOutput')

    make_new_socket(tree, name='In', io='INPUT', type='NodeSocketVector')
    # tree.inputs.new('NodeSocketVector', 'In')
    if len(functions) > 1:
        make_new_socket(tree, name='Out', io='OUTPUT', type='NodeSocketVector')
        # tree.outputs.new('NodeSocketVector', 'Out')
    else:
        make_new_socket(tree, name='OUT', io='OURPUT', type='NodeSocketFloat')
        # tree.outputs.new('NodeSocketFloat', 'Out')

    group = nodes.new(type='ShaderNodeGroup')
    group.node_tree = tree
    group_inputs.location = (length * delta, 0)

    last_out = group_inputs.outputs[0]
    for i in range(iterations):
        iterator = create_group_from_vector_function(tree_nodes,
                                                     functions,
                                                     parameters, name="iterator_" + str(i + 1))
        iterator.location = (delta, 400 - i * 200)
        iterator.hide = True
        tree_links.new(last_out, iterator.inputs[0])
        tree_links.new(group_inputs.outputs[0], iterator.inputs[1])
        last_out = iterator.outputs['Out']

    tree_links.new(last_out, group_outputs.inputs[0])
    group_outputs.location = (2 * delta, 0)

    return group


def if_node(nodes, bool_function, parameters=['True', 'False'], scalar_parameters=[], name='IfNode',
            node_type='Shader'):
    """
    creating a node that mixes two possible inputs depending on a function
    the two possible inputs are either parameters or
    scalar_parameters and denoted with 'True' and 'False'
    :param nodes:
    :param bool_function:
    :param parameters:
    :param scalar_parameters:
    :param name:
    :param node_type:
    :return:
    """
    if node_type == 'Shader':
        tree = bpy.data.node_groups.new(type='ShaderNodeTree', name=name)
    else:
        tree = bpy.data.node_groups.new(type='GeometryNodeTree', name=name)

    tree_nodes = tree.nodes
    tree_links = tree.links

    group_inputs = tree_nodes.new('NodeGroupInput')
    group_inputs.location = (0, 0)
    group_outputs = tree_nodes.new('NodeGroupOutput')

    if 'True' in parameters:
        data_type = 'Vector'
    else:
        data_type = 'Float'

    if data_type == 'Vector':
        tree.outputs.new('NodeSocketVector', 'Out')
    else:
        tree.outputs.new('NodeSocketFloat', 'Out')

    delta = 200
    length = 1

    if type == 'Shader':
        group = nodes.new(type='ShaderNodeGroup')
    else:
        group = nodes.new(type='GeometryNodeGroup')

    group.name = name
    group.node_tree = tree

    # create input nodes
    for i, parameter in enumerate(parameters):
        make_new_socket(tree, name=parameter, io='INPUT', type='NodeSocketVector')
        # tree.inputs.new('NodeSocketVector', parameter)
    for j, parameter in enumerate(scalar_parameters):
        make_new_socket(tree, name=parameter, io='INPUT', type='NodeSocketFloat')
        # tree.inputs.new('NodeSocketFloat', parameter)

    seps = []
    all_terms = bool_function.split(",")

    # find function parameters
    fcn_params = []
    mix_params = []

    for p in parameters:
        fcn_p = False
        for t in all_terms:
            if p in t:
                fcn_params.append(p)
                fcn_p = True
                break
        if not fcn_p:
            mix_params.append(p)
    fcn_scalar_params = []
    mix_scalar_params = []
    for p in scalar_parameters:
        fcn_p = False
        for t in all_terms:
            if p in t:
                fcn_scalar_params.append(p)
                fcn_p = True
                break
        if not fcn_p:
            mix_scalar_params.append(p)

    if data_type == 'Float':
        fcn = create_group_from_vector_function(tree_nodes,
                                                functions=[bool_function],
                                                parameters=fcn_params,
                                                scalar_parameters=fcn_scalar_params, name='booleFunc',
                                                node_group_type=node_type)
    else:
        fcn = create_group_from_vector_function(tree_nodes,
                                                functions=[bool_function],
                                                parameters=fcn_params,
                                                scalar_parameters=fcn_scalar_params,
                                                name='boolVecFunc', node_group_type=node_type)

    fcn.location = (length * delta, 0)
    length += 1

    for p in fcn_params + fcn_scalar_params:
        tree_links.new(group_inputs.outputs[p], fcn.inputs[p])
    if 'x' in all_terms or 'y' in all_terms or 'z' in all_terms or 'v' in all_terms:
        tree_links.new(group_inputs.outputs['In'], fcn.inputs['In'])

    mixer = create_group_from_vector_function(tree_nodes, functions=['True,x,scale,False,1,x,-,scale,add'],
                                              parameters=mix_params, scalar_parameters=mix_scalar_params,
                                              name='mixingFunction', node_group_type=node_type)
    mixer.location = (length * delta, 0)
    length += 1
    tree_links.new(fcn.outputs[0], mixer.inputs['In'])
    tree_links.new(group_inputs.outputs['True'], mixer.inputs['True'])
    tree_links.new(group_inputs.outputs['False'], mixer.inputs['False'])
    tree_links.new(mixer.outputs[0], group_outputs.inputs[0])

    group_outputs.location = (length * delta, 0)
    return group


def create_group_from_vector_function(nodes, functions, parameters=[], scalar_parameters=[], name='VectorFunction',
                                      node_group_type='Shader'):
    """
       :param scalar_parameters:
       :param nodes:
       :param node_group_type:
       :param functions: functions in polish notation for each component of the vector, "v" is refered to as a vector and "x","y","z" are the components
       :param parameters: parameters are assumed to be vectors, components are referenced with "a_x", "a_y", "a_z", etc.
       :param name: name of the group
       :return:
       """

    if node_group_type == 'Shader':
        tree = bpy.data.node_groups.new(type='ShaderNodeTree', name=name)
    else:
        tree = bpy.data.node_groups.new(type='GeometryNodeTree', name=name)

    tree_nodes = tree.nodes
    tree_links = tree.links

    delta = 200
    length = -1

    group_inputs = tree_nodes.new('NodeGroupInput')
    group_outputs = tree_nodes.new('NodeGroupOutput')

    # analyse function
    # we use standard variables v: vector, x,y,z: components of the vector
    make_new_socket(tree, name='In', io='INPUT', type='NodeSocketVector')
    # tree.inputs.new('NodeSocketVector', 'In')
    last = functions[0].split(',')[-1]
    if last in VECTOR_OPERATORS:
        last_op = 'VECTOR'
    else:
        last_op = 'VALUE'
    if len(functions) > 1:
        out_float = False
    else:
        if last_op == 'VECTOR':
            out_float = False
        else:
            out_float = True

    if out_float:
        make_new_socket(tree, name='Out', io='OUTPUT', type='NodeSocketFloat')
        # tree.outputs.new('NodeSocketFloat','Out')
    else:
        make_new_socket(tree, name='Out', io='OUTPUT', type='NodeSocketVector')
        # tree.outputs.new('NodeSocketVector','Out')

    # create component nodes if necessary
    seps = []
    all_terms = []
    for function in functions:
        all_terms += function.split(",")

    if ('x' in all_terms) or ('y' in all_terms) or ('z' in all_terms):
        sep = tree_nodes.new(type='ShaderNodeSeparateXYZ')
        tree_links.new(group_inputs.outputs['In'], sep.inputs['Vector'])
        sep.location = (length * delta, 0)
        sep.hide = True
        seps.append(sep)
    else:
        seps.append(group_inputs.outputs['In'])

    for i, parameter in enumerate(parameters):
        make_new_socket(tree, name=parameter, io='INPUT', type='NodeSocketVector')
        # tree.inputs.new('NodeSocketVector', parameter)
        # create separate xyz for each parameter if ncessary
        if (parameter + "_x" in all_terms) or (parameter + "_y" in all_terms) or (parameter + "_z" in all_terms):
            sep = tree_nodes.new(type='ShaderNodeSeparateXYZ')
            tree_links.new(group_inputs.outputs[i + 1], sep.inputs['Vector'])
            sep.location = (length * delta, -delta / 2 * (1 + i))
            sep.hide = True
            seps.append(sep)
        else:
            seps.append(group_inputs.outputs[i + 1])

    for j, parameter in enumerate(scalar_parameters):
        make_new_socket(tree, name=parameter, io='INPUT', type='NodeSocketFloat')
        # tree.inputs.new('NodeSocketFloat', parameter)
        seps.append(group_inputs.outputs[len(parameters) + j + 1])

    length -= 1

    if not out_float and last_op == 'VALUE':
        comb = tree_nodes.new(type='ShaderNodeCombineXYZ')
        tree_links.new(comb.outputs[0], group_outputs.inputs[0])

    if node_group_type == 'Shader':
        group = nodes.new(type='ShaderNodeGroup')
    else:
        group = nodes.new(type='GeometryNodeGroup')
    group.name = name
    group.node_tree = tree
    group_inputs.location = (length * delta, 0)

    stacks = []
    for function in functions:
        stacks.append(function.split(','))

    lengths = []
    for stack in stacks:
        length = 1
        for s in stack:
            if s in OPERATORS:
                length += 1
            lengths.append(length)
    # find maximal length for the various components
    length = max(lengths)
    length = int(length / 2)

    if out_float or last_op == 'VECTOR':
        build_group_component(tree, stacks[0], parameters=parameters, scalar_parameters=scalar_parameters, seps=seps,
                              inputs=group_inputs, out=group_outputs.inputs[0],
                              length=length, height=200, level=[0])
    else:
        for i, stack in enumerate(stacks):
            build_group_component(tree, stack, parameters=parameters, scalar_parameters=scalar_parameters, seps=seps,
                                  inputs=group_inputs, out=comb.inputs[i],
                                  length=length, height=i * 200, level=[0])
        length += 1
        comb.location = (length * delta, 0)

    length += 1
    group_outputs.location = (length * delta, 0)

    remove_unlinked_nodes(tree)
    return group


def create_group_from_scalar_function(nodes, functions, parameters=[], vector_parameters=[], name='ScalarFunction',
                                      node_group_type='Shader', output_names=[]):
    """
       :param node_group_type:
       :param nodes:
       :param functions: functions in polish notation for a scalar, "x" is the input
       :param parameters: parameters are assumed to be scalars
       :param name: name of the group
       :return:
       """

    if node_group_type == 'Shader':
        tree = bpy.data.node_groups.new(type='ShaderNodeTree', name=name)
        group = nodes.new(type='ShaderNodeGroup')
    else:
        tree = bpy.data.node_groups.new(type='GeometryNodeTree', name=name)
        group = nodes.new(type='GeometryNodeGroup')

    tree_nodes = tree.nodes
    tree_links = tree.links

    delta = 200
    length = -1

    group_inputs = tree_nodes.new('NodeGroupInput')
    group_outputs = tree_nodes.new('NodeGroupOutput')

    # analyse function
    make_new_socket(tree, name='In', io='INPUT', type='NodeSocketFloat')

    for i in range(len(functions)):
        if len(output_names) > i:
            out = output_names[i]
        else:
            out = 'Out' + str(i + 1)
        make_new_socket(tree, name=out, io='OUTPUT', type='NodeSocketFloat')

    for i, parameter in enumerate(parameters):
        make_new_socket(tree, name=parameter, io='INPUT', type='NodeSocketFloat')

    length -= 1

    group.name = name
    group.node_tree = tree
    group_inputs.location = (length * delta, 0)

    stacks = []
    for function in functions:
        stacks.append(function.split(','))

    lengths = []
    for stack in stacks:
        length = 1
        for s in stack:
            if s in OPERATORS:
                length += 1
            lengths.append(length)
    # find maximal length for the various components
    length = max(lengths)
    length = int(length / 2)

    for i, stack in enumerate(stacks):
        build_scalar_group_component(tree, stack, parameters=parameters,
                                     inputs=group_inputs, out=group_outputs.inputs[i],
                                     length=length, height=i * 200, level=[0])
    length += 1

    length += 1
    group_outputs.location = (length * delta, 0)

    remove_unlinked_nodes(tree)
    return group


def build_group_component(tree, stack, parameters=[], scalar_parameters=[], seps=None, inputs=None,
                          out=None, length=1, unary=None, last_operator=None, height=0, level=[0]):
    """
    there is a subtlety with VectorMath nodes, they always carry two outputs. The first one is 'Vector' and the second one is 'Value'
    there is more work to be done, to do this correctly, so far there is only a workaround to incorporate the 'LENGTH' operation, which yields a scalar output
    :param level: captures the structure of the tree to place the node at the right location
    :param scalar_parameters:
    :param tree:
    :param stack:
    :param parameters:
    :param seps:
    :param inputs:
    :param out:
    :param length:
    :param unary:
    :param last_operator:
    :param height:
    :return:
    """

    left_empty = True
    if unary:
        right_empty = False
    else:
        right_empty = True

    new_node_math = None

    while (left_empty or right_empty) and len(stack) > 0:
        next_element = stack.pop()
        if next_element in OPERATORS:
            # warning not all are implemented yet
            # always implemented as needed
            unary = False
            if next_element == '*':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'MULTIPLY'
            elif next_element == 'mul':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'MULTIPLY'
            elif next_element == '%':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'MODULO'
            elif next_element == 'mod':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'MODULO'
            elif next_element == '/':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'DIVIDE'
            elif next_element == 'div':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'DIVIDE'
            elif next_element == '+':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'ADD'
            elif next_element == 'add':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'ADD'
            elif next_element == 'sub':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'SUBTRACT'
            elif next_element == '-':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'SUBTRACT'
            elif next_element == '<':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'LESS_THAN'
            elif next_element == '>':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'GREATER_THAN'
            elif next_element == '=':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'COMPARE'
                new_node_math.inputs[2].default_value = 0
            elif next_element == 'min':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'MINIMUM'
            elif next_element == 'max':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'MAXIMUM'
            elif next_element == 'sin':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'SINE'
                unary = True
            elif next_element == 'cos':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'COSINE'
                unary = True
            elif next_element == 'tan':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'TANGENT'
                unary = True
            elif next_element == 'atan2':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'ARCTAN2'
                unary = False
            elif next_element == 'round':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'ROUND'
                unary = True
            elif next_element == 'abs':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'ABSOLUTE'
                unary = True
            elif next_element == 'round':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'ROUND'
                unary = True
            elif next_element == 'floor':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'FLOOR'
                unary = True
            elif next_element == 'vfloor':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'FLOOR'
                unary = True
            elif next_element == 'ceil':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'CEIL'
                unary = True
            elif next_element == 'length':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'LENGTH'
                unary = True
            elif next_element == 'sqrt':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'SQRT'
                unary = True
            elif next_element == 'scale':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'SCALE'
            elif next_element == 'cross':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'CROSS_PRODUCT'
            elif next_element == 'dot':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'DOT_PRODUCT'

            # y-position and logging
            o = ""
            for i in range(length):
                o += '\t'
            o += next_element

            if unary:
                new_level = level + [0]
            else:
                if right_empty:
                    new_level = level + [-1]
                else:
                    new_level = level[0:-1] + [1]
            y_pos = 0
            for i, bit in enumerate(new_level):
                y_pos += bit * 500 / (i + 1)

            new_node_math.location = (length * 200, y_pos)
            new_node_math.hide = True

            print(o)

            if last_operator is None:
                # link first operator to the output
                tree.links.new(new_node_math.outputs[0], out)
            elif right_empty:
                # make sure that the type fits, e.g. the operator 'LENGTH' first has an output of type 'VECTOR'
                success = False
                for o in new_node_math.outputs:
                    if o.type == last_operator.inputs[1].type:
                        tree.links.new(o, last_operator.inputs[1])
                        success = True
                        break
                if not success:
                    # try the other way round
                    for i in range(len(last_operator.inputs) - 1, -1, -1):
                        if last_operator.inputs[i].type == new_node_math.outputs[0].type:
                            tree.links.new(new_node_math.outputs[0], last_operator.inputs[i])
                            break
                right_empty = False
            elif left_empty:
                # make sure that the type fits, e.g. the operator 'LENGTH' first has an output of type 'VECTOR'
                success = False
                for o in new_node_math.outputs:
                    if o.type == last_operator.inputs[0].type:
                        tree.links.new(o, last_operator.inputs[0])
                        success = True
                        break
                if not success:
                    for i in range(len(last_operator.inputs)):
                        if last_operator.inputs[i].type == new_node_math.outputs[0].type:
                            tree.links.new(new_node_math.outputs[0], last_operator.inputs[i])
                            break
                left_empty = False

        elif next_element == 'v':
            if last_operator is None:
                tree.links.new(seps[0], out)
            elif right_empty:
                tree.links.new(seps[0], last_operator.inputs[1])
                right_empty = False
            elif left_empty:
                tree.links.new(seps[0], last_operator.inputs[0])
                left_empty = False
        elif next_element == 'x':
            if last_operator is None:
                tree.links.new(seps[0].outputs[0], out)
            elif right_empty:
                if last_operator.operation == 'SCALE':
                    o = last_operator.inputs[3]
                else:
                    o = last_operator.inputs[1]
                tree.links.new(seps[0].outputs[0], o)
                right_empty = False
            elif left_empty:
                tree.links.new(seps[0].outputs[0], last_operator.inputs[0])
                left_empty = False
        elif next_element == 'y':
            if last_operator is None:
                tree.links.new(seps[0].outputs[1], out)
            elif right_empty:
                tree.links.new(seps[0].outputs[1], last_operator.inputs[1])
                right_empty = False
            elif left_empty:
                tree.links.new(seps[0].outputs[1], last_operator.inputs[0])
                left_empty = False
        elif next_element == 'z':
            if last_operator is None:
                tree.links.new(seps[0].outputs[2], out)
            elif right_empty:
                tree.links.new(seps[0].outputs[2], last_operator.inputs[1])
                right_empty = False
            elif left_empty:
                tree.links.new(seps[0].outputs[2], last_operator.inputs[0])
                left_empty = False
        # check for vector parameter access
        elif next_element in parameters:
            pos = parameters.index(next_element)
            if last_operator is None:
                tree.links.new(seps[pos + 1], out)
            elif right_empty:
                tree.links.new(seps[pos + 1], last_operator.inputs[1])
                right_empty = False
            elif left_empty:
                tree.links.new(seps[pos + 1], last_operator.inputs[0])
                left_empty = False
        # check for scalar parameter access
        elif next_element in scalar_parameters:
            pos = scalar_parameters.index(next_element)
            vecs = len(parameters)
            if last_operator is None:
                tree.links.new(seps[pos + vecs + 1], out)
            elif right_empty:
                tree.links.new(seps[pos + vecs + 1], last_operator.inputs[1])
                right_empty = False
            elif left_empty:
                tree.links.new(seps[pos + vecs + 1], last_operator.inputs[0])
                left_empty = False
        # remove _x, _y, _z flag for parameter detection
        elif next_element[0:-2] in parameters:
            pos = parameters.index(next_element[0:-2])
            comp = {'x': 0, 'y': 1, 'z': 2}[next_element[-1:]]
            if last_operator is None:
                tree.links.new(seps[pos + 1].outputs[comp], out)
            elif right_empty:
                tree.links.new(seps[pos + 1].outputs[comp], last_operator.inputs[1])
                right_empty = False
            elif left_empty:
                tree.links.new(seps[pos + 1].outputs[comp], last_operator.inputs[0])
                left_empty = False
        # check for simple numbers
        else:
            if next_element == 'pi':
                number = np.pi
            elif next_element[0] == '(':
                next_element = next_element[1:-1]
                numbers = next_element.split(' ')
                vals = []
                for i in range(len(numbers)):
                    if numbers[i] == 'pi':
                        vals.append(np.pi)
                    else:
                        vals.append(float(numbers[i]))
                number = Vector(vals)
            else:
                number = float(next_element)
            if last_operator is None:
                out.default_value = number
            elif right_empty:
                last_operator.inputs[1].default_value = number
                right_empty = False
            elif left_empty:
                last_operator.inputs[0].default_value = number
                left_empty = False
            else:
                raise "Something went wrong. The number " + next_element + " is left over."

        # if a new operator is processed the function has to be called again
        if new_node_math:
            build_group_component(tree, stack, parameters=parameters, scalar_parameters=scalar_parameters, seps=seps,
                                  inputs=inputs, out=out,
                                  length=length - 1, unary=unary, last_operator=new_node_math, height=height,
                                  level=new_level)
            new_node_math = None


def build_scalar_group_component(tree, stack, parameters=[], vector_parameters=[], inputs=None,
                                 out=None, length=1, unary=None, last_operator=None, height=0, level=[0]):
    """
    :param level: captures the structure of the tree to place the node at the right location
    :param tree:
    :param stack:
    :param parameters:
    :param seps:
    :param inputs:
    :param out:
    :param length:
    :param unary:
    :param last_operator:
    :param height:
    :return:
    """

    left_empty = True
    if unary:
        right_empty = False
    else:
        right_empty = True

    new_node_math = None

    while (left_empty or right_empty) and len(stack) > 0:
        next_element = stack.pop()
        if next_element in OPERATORS:
            # warning not all are implemented yet
            # always implemented as needed
            unary = False
            if next_element == '*':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'MULTIPLY'
            elif next_element == 'mul':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'MULTIPLY'
            elif next_element == '%':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'MODULO'
            elif next_element == 'mod':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'MODULO'
            elif next_element == '/':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'DIVIDE'
            elif next_element == 'div':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'DIVIDE'
            elif next_element == '+':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'ADD'
            elif next_element == 'add':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'ADD'
            elif next_element == 'sub':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'SUBTRACT'
            elif next_element == '-':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'SUBTRACT'
            elif next_element == '<':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'LESS_THAN'
            elif next_element == '>':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'GREATER_THAN'
            elif next_element == '=':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'COMPARE'
                new_node_math.inputs[2].default_value = 0
            elif next_element == 'min':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'MINIMUM'
            elif next_element == 'max':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'MAXIMUM'
            elif next_element == 'sin':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'SINE'
                unary = True
            elif next_element == 'cos':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'COSINE'
                unary = True
            elif next_element == 'tan':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'TANGENT'
                unary = True
            elif next_element == 'atan2':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'ARCTAN2'
                unary = False
            elif next_element == 'round':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'ROUND'
                unary = True
            elif next_element == 'abs':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'ABSOLUTE'
                unary = True
            elif next_element == 'round':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'ROUND'
                unary = True
            elif next_element == 'floor':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'FLOOR'
                unary = True
            elif next_element == 'vfloor':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'FLOOR'
                unary = True
            elif next_element == 'ceil':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'CEIL'
                unary = True
            elif next_element == 'length':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'LENGTH'
                unary = True
            elif next_element == 'sqrt':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'SQRT'
                unary = True
            elif next_element == 'scale':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'SCALE'
            elif next_element == 'cross':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'CROSS_PRODUCT'
            elif next_element == 'dot':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'DOT_PRODUCT'

            # y-position and logging
            o = ""
            for i in range(length):
                o += '\t'
            o += next_element

            if unary:
                new_level = level + [0]
            else:
                if right_empty:
                    new_level = level + [-1]
                else:
                    new_level = level[0:-1] + [1]
            y_pos = 0
            for i, bit in enumerate(new_level):
                y_pos += bit * 500 / (i + 1)
            new_node_math.location = (length * 200, y_pos)
            new_node_math.hide = True

            print(o)

            if last_operator is None:
                # link first operator to the output
                tree.links.new(new_node_math.outputs[0], out)
            elif right_empty:
                # make sure that the type fits, e.g. the operator 'LENGTH' first has an output of type 'VECTOR'
                success = False
                for o in new_node_math.outputs:
                    if o.type == last_operator.inputs[1].type:
                        tree.links.new(o, last_operator.inputs[1])
                        success = True
                        break
                if not success:
                    # try the other way round
                    for i in range(len(last_operator.inputs) - 1, -1, -1):
                        if last_operator.inputs[i].type == new_node_math.outputs[0].type:
                            tree.links.new(new_node_math.outputs[0], last_operator.inputs[i])
                            break
                right_empty = False
            elif left_empty:
                # make sure that the type fits, e.g. the operator 'LENGTH' first has an output of type 'VECTOR'
                success = False
                for o in new_node_math.outputs:
                    if o.type == last_operator.inputs[0].type:
                        tree.links.new(o, last_operator.inputs[0])
                        success = True
                        break
                if not success:
                    for i in range(len(last_operator.inputs)):
                        if last_operator.inputs[i].type == new_node_math.outputs[0].type:
                            tree.links.new(new_node_math.outputs[0], last_operator.inputs[i])
                            break
                left_empty = False

        elif next_element == 'x':
            if last_operator is None:
                tree.links.new(inputs.outputs[0], out)
            elif right_empty:
                if last_operator.operation == 'SCALE':
                    o = last_operator.inputs[3]
                else:
                    o = last_operator.inputs[1]
                tree.links.new(inputs.outputs[0], o)
                right_empty = False
            elif left_empty:
                tree.links.new(inputs.outputs[0], last_operator.inputs[0])
                left_empty = False
        # check for scalar parameter access
        elif next_element in parameters:
            pos = parameters.index(next_element)
            if last_operator is None:
                tree.links.new(inputs.outputs[pos + 1], out)
            elif right_empty:
                tree.links.new(inputs.outputs[pos + 1], last_operator.inputs[1])
                right_empty = False
            elif left_empty:
                tree.links.new(inputs.outputs[pos + 1], last_operator.inputs[0])
                left_empty = False
            # check for scalar parameter access
            elif next_element in vector_parameters:
                pos = parameters.index(next_element)
                if last_operator is None:
                    tree.links.new(inputs.outputs[pos + 1], out)
                elif right_empty:
                    tree.links.new(inputs.outputs[pos + 1], last_operator.inputs[1])
                    right_empty = False
                elif left_empty:
                    tree.links.new(inputs.outputs[pos + 1], last_operator.inputs[0])
                    left_empty = False
        # check for simple numbers
        else:
            if next_element == 'pi':
                number = np.pi
            elif next_element[0] == '(':
                next_element = next_element[1:-1]
                numbers = next_element.split(' ')
                vals = []
                for i in range(len(numbers)):
                    if numbers[i] == 'pi':
                        vals.append(np.pi)
                    else:
                        vals.append(float(numbers[i]))
                number = Vector(vals)
            else:
                number = float(next_element)
            if last_operator is None:
                out.default_value = number
            elif right_empty:
                last_operator.inputs[1].default_value = number
                right_empty = False
            elif left_empty:
                last_operator.inputs[0].default_value = number
                left_empty = False
            else:
                raise "Something went wrong. The number " + next_element + " is left over."

        # if a new operator is processed the function has to be called again
        if new_node_math:
            build_scalar_group_component(tree, stack, parameters=parameters, vector_parameters=vector_parameters,
                                         inputs=inputs, out=out,
                                         length=length - 1, unary=unary, last_operator=new_node_math, height=height,
                                         level=new_level)
            new_node_math = None


def make_new_socket(tree, name='mySocket', io='INPUT', type='NodeSocketFloat'):
    """

    :param tree:
    :param name:
    :param io:
    :param type: select one of the following sockets
    ('NodeSocketString', 'NodeSocketBool', 'NodeSocketMaterial', 'NodeSocketVector', 'NodeSocketInt', 'NodeSocketGeometry', 'NodeSocketCollection', 'NodeSocketTexture', 'NodeSocketFloat', 'NodeSocketColor', 'NodeSocketObject', 'NodeSocketRotation', 'NodeSocketImage')
    :return:
    """
    version = bpy.app.version_string
    if int(version[0]) < 4:
        if io == 'INPUT':
            tree.inputs.new(type, 'name')
        else:
            tree.outputs.new(type, 'name')
    else:
        tree.interface.new_socket(name=name, in_out=io, socket_type=type)


def remove_unlinked_nodes(node_tree):
    linked_nodes = set()

    for link in node_tree.links:
        linked_nodes.add(link.from_node)
        linked_nodes.add(link.to_node)

    unlinked_nodes = set(node_tree.nodes) - linked_nodes
    while unlinked_nodes:
        node_tree.nodes.remove(unlinked_nodes.pop())


def create_shader_group_from_function(nodes, function, parameters=[], inputType='FLOAT', outputType='FLOAT',
                                      name="CustomGroup"):
    """

    :param function: function in polish notation
    :param parameters: list of parameters that appear in the function, there is a socket created for every parameter, so far parameters are assumed to be floats
    :param inputType: type of input, eg. 'FLOAT', 'VECTOR', 'COLOR'
    :param outputType: type of output, eg. 'FLOAT', 'VECTOR', 'COLOR'
    :param name: name of the group
    :return:
    """

    # container for the group

    tree = bpy.data.node_groups.new(name, 'ShaderNodeTree')
    tree_nodes = tree.nodes
    tree_links = tree.links
    delta = 200
    length = 1

    group_inputs = tree_nodes.new('NodeGroupInput')
    group_outputs = tree_nodes.new('NodeGroupOutput')

    # analyse function
    # we use standard variables v: vector, x,y,z: components of the vector
    # parsing:

    # create sockets
    if inputType == 'FLOAT':
        type = 'NodeSocketFloat'
    elif inputType == 'VECTOR':
        type = 'NodeSocketVector'
    elif inputType == 'COLOR':
        type = 'NodeSocketColor'
    make_new_socket(tree, name='In', io='INPUT', type=type)

    if outputType == 'FLOAT':
        type = 'NodeSocketFloat'
    elif outputType == 'VECTOR':
        type = 'NodeSocketVector'
    elif outputType == 'COLOR':
        type = 'NodeSocketColor'
    make_new_socket(tree, name='Out', io='OUTPUT', type=type)

    for parameter in parameters:
        make_new_socket(tree, name=parameter, io='INPUT', type='NodeSocketFloat')
        # tree.inputs.new('NodeSocketFloat', parameter)

    stack = function.split(',')
    for s in stack:
        if s in OPERATORS:
            length += 1

    group = nodes.new(type='ShaderNodeGroup')
    group.node_tree = tree
    group_inputs.location = (-length * delta, 0)
    length -= 1

    sep = None
    if 'x' in stack or 'y' in stack or 'z' in stack:
        # create separate xyz
        sep = tree_nodes.new(type='ShaderNodeSeparateXYZ')
        tree_links.new(group_inputs.outputs['In'], sep.inputs['Vector'])
        sep.location = (-length * delta, 0)
        length -= 1
    # tree is built from right to left therefore the length is 0 initially
    build_group(tree, stack, parameters=parameters, sep=sep, input=group_inputs, output=group_outputs, length=0)

    length += 1
    group_outputs.location = (length * delta, 0)

    return group


def build_group(tree, stack, parameters=[], sep=None, input=None,
                output=None, length=1, unary=False, last_operator=None):
    """
    deprecated use build_group_component
    """
    left_empty = True
    if unary:
        right_empty = False
    else:
        right_empty = True

    new_node_math = None
    while (left_empty or right_empty) and len(stack) > 0:
        next_element = stack.pop()
        out = ""
        for i in range(length):
            out += '\t'
        out += next_element
        print(out)
        if next_element in OPERATORS:
            # warning not all are implemented yet
            # always implemented as needed
            unary = False
            new_node_math = tree.nodes.new(type='ShaderNodeMath')
            new_node_math.location = (-length * 200, (-1) ** length * 100)
            if next_element == '*':
                new_node_math.operation = 'MULTIPLY'
            elif next_element == '/':
                new_node_math.operation = 'DIVIDE'
            elif next_element == '+':
                new_node_math.operation = 'ADD'
            elif next_element == '-':
                new_node_math.operation = 'SUBTRACT'
            elif next_element == '<':
                new_node_math.operation = 'LESS_THAN'
            elif next_element == '>':
                new_node_math.operation = 'GREATER_THAN'
            elif next_element == 'sin':
                new_node_math.operation = 'SINE'
                unary = True
            elif next_element == 'cos':
                new_node_math.operation = 'COSINE'
                unary = True
            elif next_element == 'tan':
                new_node_math.operation = 'TANGENT'
                unary = True
            elif next_element == 'round':
                new_node_math.operation = 'ROUND'
                unary = True
            elif next_element == 'abs':
                new_node_math.operation = 'ABSOLUTE'
                unary = True
            elif next_element == 'round':
                new_node_math.operation = 'ROUND'
                unary = True
            elif next_element == 'floor':
                new_node_math.operation = 'FLOOR'
                unary = True
            elif next_element == 'ceil':
                new_node_math.operation = 'CEIL'
                unary = True
            if last_operator is None:
                tree.links.new(new_node_math.outputs[0], output.inputs[0])
            elif right_empty:
                tree.links.new(new_node_math.outputs[0], last_operator.inputs[1])
                right_empty = False
            elif left_empty:
                tree.links.new(new_node_math.outputs[0], last_operator.inputs[0])
                left_empty = False
        elif next_element == 'x':
            if last_operator is None:
                tree.links.new(sep.outputs[0], output.inputs[0])
            if right_empty:
                tree.links.new(sep.outputs[0], last_operator.inputs[1])
                right_empty = False
            elif left_empty:
                tree.links.new(sep.outputs[0], last_operator.inputs[0])
                left_empty = False
        elif next_element == 'y':
            if last_operator is None:
                tree.links.new(sep.outputs[1], output.inputs[0])
            if right_empty:
                tree.links.new(sep.outputs[1], last_operator.inputs[1])
                right_empty = False
            elif left_empty:
                tree.links.new(sep.outputs[1], last_operator.inputs[0])
                left_empty = False
        elif next_element == 'z':
            if last_operator is None:
                tree.links.new(sep.outputs[2], output.inputs[0])
            if right_empty:
                tree.links.new(sep.outputs[2], last_operator.inputs[1])
                right_empty = False
            elif left_empty:
                tree.links.new(sep.outputs[2], last_operator.inputs[0])
                left_empty = False
        elif next_element in parameters:
            pos = parameters.index(next_element)
            if last_operator is None:
                tree.links.new(input.outputs[pos + 1], output.inputs[0])
            if right_empty:
                tree.links.new(input.outputs[pos + 1], last_operator.inputs[1])
                right_empty = False
            elif left_empty:
                tree.links.new(input.outputs[pos + 1], last_operator.inputs[0])
                left_empty = False
        else:
            if next_element == 'pi':
                number = np.pi
            else:
                number = float(next_element)
            if right_empty:
                last_operator.inputs[1].default_value = number
                right_empty = False
            elif left_empty:
                last_operator.inputs[0].default_value = number
                left_empty = False
            else:
                raise "Something went wrong. The number " + next_element + " is left over."

        if new_node_math:  # if a new operator is processed the function has to be called again
            build_group(tree, stack, parameters=parameters, sep=sep, input=input, output=output,
                        length=length + 1, unary=unary, last_operator=new_node_math)
            new_node_math = None


def create_shader_from_function(material, hue_functions, scale=[1, 1, 1], emission_strength=0.3,
                                input='geometry_position'):
    """
    this is an unskillful way to parse the function, provided in inverse polish notation and create the corresponding
    math nodes for the texture in blender

    :param material:
    :param hue_functions:
    :param scale:
    :param emission_strength:
    :return:
    """
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    node_bsdf = nodes['Principled BSDF']
    node_bsdf.inputs['Emission Strength'].default_value = emission_strength

    dist = 10
    # prepare the calculation of the phase from the imaginary and real part of the resulting complex function

    if input == 'tex_coord_obj':
        input_coords = nodes.new(type='ShaderNodeTexCoord')
        out = input_coords.outputs['Object']
    else:
        input_coords = nodes.new(type='ShaderNodeNewGeometry')
        out = input_coords.outputs['Position']
    input_coords.location = (-(dist + 2) * 200, 0)
    node_scale = nodes.new(type='ShaderNodeVectorMath')
    node_scale.operation = 'DIVIDE'
    node_scale.location = (-(dist + 1) * 200, 0)
    if len(scale) == 2:
        scale.append(1)
    node_scale.inputs[1].default_value = scale
    links.new(out, node_scale.inputs[0])

    node_sep = nodes.new(type='ShaderNodeSeparateXYZ')
    node_sep.location = (-dist * 200, 0)
    links.new(node_scale.outputs['Vector'], node_sep.inputs['Vector'])

    out_x = node_sep.outputs['X']
    out_y = node_sep.outputs['Y']
    out_z = node_sep.outputs['Z']
    number_of_shadings = int(len(hue_functions) / 2)
    dialer = []
    dialer_nodes = []
    hues = []

    for s in range(0, number_of_shadings):
        re_function = hue_functions[2 * s]
        im_function = hue_functions[2 * s + 1]

        # create the real part and imaginary part of the function
        node_re_out = create_part(re_function, nodes, links, out_x, out_y, out_z, x_loc=-200, y_loc=-200 + 400 * s)
        node_im_out = create_part(im_function, nodes, links, out_x, out_y, out_z, x_loc=-200, y_loc=+200 + 400 * s)

        node_arctan2 = nodes.new(type='ShaderNodeMath')
        node_arctan2.location = (0, 0 + 400 * s)
        node_arctan2.operation = 'ARCTAN2'
        links.new(node_im_out, node_arctan2.inputs[0])
        links.new(node_re_out, node_arctan2.inputs[1])

        node_shift_pi = nodes.new(type='ShaderNodeMath')
        node_shift_pi.operation = 'ADD'
        node_shift_pi.location = (200, 0 + 400 * s)
        node_shift_pi.inputs[1].default_value = np.pi
        links.new(node_arctan2.outputs[0], node_shift_pi.inputs[0])

        node_div_2pi = nodes.new(type='ShaderNodeMath')
        node_div_2pi.operation = 'DIVIDE'
        node_div_2pi.location = (2 * 200, 0 + 400 * s)
        node_div_2pi.inputs[1].default_value = 2 * np.pi
        links.new(node_shift_pi.outputs[0], node_div_2pi.inputs[0])

        node_hue = nodes.new(type='ShaderNodeHueSaturation')
        node_hue.location = (3 * 200, 0 + 400 * s)
        node_hue.inputs['Color'].default_value = [1, 0, 0, 1]
        links.new(node_div_2pi.outputs[0], node_hue.inputs['Hue'])

        hues.append(node_hue)

    # merge all hue_nodes into the bsdf with mixrgb nodes.

    for i, hue in enumerate(hues):
        if (i + 1) < len(hues):
            # create a rgb mixer for the next color_layer
            mixer = nodes.new(type='ShaderNodeMixRGB')
            mixer.inputs[0].default_value = 0
            dialer.append(mixer.inputs[0])
            mixer.location = (4 * 200, 400 * i)
            links.new(hue.outputs[0], mixer.inputs[1])
            if len(dialer_nodes) > 0:
                links.new(mixer.outputs['Color'], dialer_nodes[-1].inputs[2])
            dialer_nodes.append(mixer)
        else:
            if len(dialer_nodes) > 0:
                links.new(hue.outputs['Color'], dialer_nodes[-1].inputs[2])

    # link the hue nodes and mixers to the principled bsdf
    if len(dialer_nodes) == 0:
        # only one hue_node
        links.new(hues[0].outputs['Color'], node_bsdf.inputs['Base Color'])
        links.new(hues[0].outputs['Color'], node_bsdf.inputs[EMISSION])
    else:
        links.new(dialer_nodes[0].outputs['Color'], node_bsdf.inputs['Base Color'])
        links.new(dialer_nodes[0].outputs['Color'], node_bsdf.inputs[EMISSION])

    node_bsdf.location = (4 * 200 + 100, 0)
    nodes['Material Output'].location = (6 * 200, 0)

    return dialer


def create_part(function, nodes, links, out_x, out_y, out_z, x_loc=0, y_loc=0):
    """
    This method systematically builds the node tree from the function.
    First a stack is created to process the entries from last to first.
    The function should be a post-order traverse of the tree that will be built
    :param function:
    :param nodes:
    :param links:
    :param out_x:
    :param out_y:
    :param x_loc:
    :param y_loc:
    :return:
    """
    stack = function.split(',')
    if len(stack) == 1:
        # trivial identity function
        first = stack.pop()
        if first == 'x':
            return out_x
        elif first == 'y':
            return out_y
        elif first == 'z':
            return out_z
    else:
        # first element must be an operation
        element = stack.pop()
        root_node_math = nodes.new('ShaderNodeMath')
        root_node_math.location = (x_loc, y_loc)
        unary = False

        if element == '*':
            root_node_math.operation = 'MULTIPLY'
        elif element == '/':
            root_node_math.operation = 'DIVIDE'
        elif element == '+':
            root_node_math.operation = 'ADD'
        elif element == '-':
            root_node_math.operation = 'SUBTRACT'
        elif element == 'sin':
            root_node_math.operation = 'SINE'
            unary = True

        build_recursively(stack, root_node_math, nodes, links, out_x, out_y, out_z, x_loc=x_loc - 200, y_loc=y_loc,
                          unary=unary)

    # return the output of the lowest level node
    return root_node_math.outputs[0]


def build_recursively(stack, node_math, nodes, links, out_x, out_y, out_z, x_loc=0, y_loc=0, unary=False):
    left_empty = True
    if unary:  # in case of an unary operator the right branch is deactivated
        right_empty = False
    else:
        right_empty = True

    if len(stack) == 0:
        pass
    else:
        while left_empty or right_empty:
            next_element = stack.pop()
            if next_element in ['*', '/', '+', '-', 'sin']:
                unary = False
                new_node_math = nodes.new(type='ShaderNodeMath')
                if next_element == '*':
                    new_node_math.operation = 'MULTIPLY'
                elif next_element == '/':
                    new_node_math.operation = 'DIVIDE'
                elif next_element == '+':
                    new_node_math.operation = 'ADD'
                elif next_element == '-':
                    new_node_math.operation = 'SUBTRACT'
                elif next_element == 'sin':
                    new_node_math.operation = 'SINE'
                    unary = True
                if right_empty:
                    links.new(new_node_math.outputs[0], node_math.inputs[1])
                    new_node_math.location = (x_loc, y_loc - 100)
                    right_empty = False
                elif left_empty:
                    links.new(new_node_math.outputs[0], node_math.inputs[0])
                    new_node_math.location = (x_loc, y_loc + 100)
                    left_empty = False
                build_recursively(stack, new_node_math, nodes, links, out_x, out_y, out_z, x_loc=x_loc - 200,
                                  y_loc=y_loc,
                                  unary=unary)
            elif next_element == 'x':
                if right_empty:
                    links.new(out_x, node_math.inputs[1])
                    right_empty = False
                elif left_empty:
                    links.new(out_x, node_math.inputs[0])
                    left_empty = False
            elif next_element == 'y':
                if right_empty:
                    links.new(out_y, node_math.inputs[1])
                    right_empty = False
                elif left_empty:
                    links.new(out_y, node_math.inputs[0])
                    left_empty = False
            elif next_element == 'z':
                if right_empty:
                    links.new(out_z, node_math.inputs[1])
                    right_empty = False
                elif left_empty:
                    links.new(out_z, node_math.inputs[0])
                    left_empty = False
            else:
                if next_element == 'pi':
                    number = np.pi
                else:
                    number = float(next_element)
                if right_empty:
                    node_math.inputs[1].default_value = number
                    right_empty = False
                elif left_empty:
                    node_math.inputs[0].default_value = number
                    left_empty = False
                else:
                    raise "Something went wrong. The number " + next_element + " is left over."


#####################
# volume materials  #
#####################

def set_volume_scatter(bob, value, begin_time=0):
    obj = get_obj(bob)
    frame = begin_time * FRAME_RATE
    material = obj.data.materials[0]
    if material:
        nodes = material.node_tree.nodes
        scatter_node = nodes["Volume Scatter"]
        if scatter_node:
            density_socket = scatter_node.inputs['Density']
            density_socket.default_value = value
            insert_keyframe(density_socket, 'default_value', frame)


def change_volume_scatter(bob, final_value, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
    frame = begin_time * FRAME_RATE
    obj = get_obj(bob)

    material = obj.data.materials[0]
    if material:
        nodes = material.node_tree.nodes
        scatter_node = nodes["Volume Scatter"]
        if scatter_node:
            density_socket = scatter_node.inputs['Density']

        if frame > 0:
            set_frame(frame - 1)
            value_old = density_socket.default_value
        else:
            value_old = 0
    density_socket.default_value = value_old
    insert_keyframe(density_socket, 'default_value', frame - 1)
    density_socket.default_value = final_value
    insert_keyframe(density_socket, 'default_value', frame + int(transition_time * FRAME_RATE))


def set_volume_absorption_of_material(material, value):
    if material:
        nodes = material.node_tree.nodes
        scatter_node = nodes["Volume Absorption"]
        if scatter_node:
            density_socket = scatter_node.inputs['Density']
            density_socket.default_value = value
        else:
            scatter_node = nodes.new(type="ShaderNodeVolumeAbsorption")
            scatter_node.inputs['Density'].default_value = value
            material.node_tree.links.new(scatter_node.outputs[0], nodes.get('Material Output').inputs['Volume'])


def set_volume_scatter_of_material(material, value):
    if material:
        nodes = material.node_tree.nodes
        scatter_node = nodes.get("Volume Scatter")
        if scatter_node:
            density_socket = scatter_node.inputs['Density']
            density_socket.default_value = value
        else:
            scatter_node = nodes.new(type="ShaderNodeVolumeScatter")
            scatter_node.inputs['Density'].default_value = value
            material.node_tree.links.new(scatter_node.outputs[0], nodes.get('Material Output').inputs['Volume'])


def set_volume_absorption(bob, value, begin_time=0):
    obj = get_obj(bob)
    frame = begin_time * FRAME_RATE
    material = obj.data.materials[0]
    if material:
        nodes = material.node_tree.nodes
        scatter_node = nodes["Volume Absorption"]
        if scatter_node:
            density_socket = scatter_node.inputs['Density']
            density_socket.default_value = value
            insert_keyframe(density_socket, 'default_value', frame)


def change_volume_absorption(bob, final_value, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
    frame = begin_time * FRAME_RATE
    obj = get_obj(bob)

    material = obj.data.materials[0]
    if material:
        nodes = material.node_tree.nodes
        scatter_node = nodes["Volume Absorption"]
        if scatter_node:
            density_socket = scatter_node.inputs['Density']

        if frame > 0:
            set_frame(frame - 1)
            value_old = density_socket.default_value
        else:
            value_old = 0
    density_socket.default_value = value_old
    insert_keyframe(density_socket, 'default_value', frame - 1)
    density_socket.default_value = final_value
    insert_keyframe(density_socket, 'default_value', frame + int(transition_time * FRAME_RATE))


#####################
# work with splines #
#####################

def merge_splines(b_objects):
    bob0 = None
    obj0 = None
    for i, bob in enumerate(b_objects):
        if i == 0:
            bob0 = bob
            obj0 = bob0.ref_obj
            loc = Vector(obj0.location[0:3])
            for spline in obj0.data.splines:
                for p in spline.bezier_points:
                    p.co += loc
                    p.handle_left += loc
                    p.handle_right += loc
        else:
            obj = bob.ref_obj
            loc = Vector(obj.location[0:3])
            for spline in get_splines(obj):
                ns = add_bezier_spline(obj0.data.splines, len(spline.bezier_points) - 1,
                                       True)  # -1, because one point is there already
                for p_dest, p_src in zip(ns.bezier_points, spline.bezier_points):
                    p_dest.co = Vector(p_src.co[0:3]) + loc
                    p_dest.handle_left_type = p_src.handle_left_type
                    p_dest.handle_left = Vector(p_src.handle_left[0:3]) + loc
                    p_dest.handle_right_type = p_src.handle_right_type
                    p_dest.handle_right = Vector(p_src.handle_right[0:3]) + loc
    return bob0


def get_spline_bounding_box(b_obj):
    """
    returns the bounding box for curve of splines with Bezier points

    :param b_obj:
    :return:
    """
    obj = get_obj(b_obj)
    data = obj.data

    mi = [np.inf, np.inf, np.inf]
    ma = [-np.inf, -np.inf, -np.inf]
    for spline in data.splines:
        for p in spline.bezier_points:
            for i, v in enumerate(p.co):
                if mi[i] > v:
                    mi[i] = v
                if ma[i] < v:
                    ma[i] = v

    # translate to the curve location
    for i in range(3):
        mi[i] += obj.location[i]
        ma[i] += obj.location[i]

    return BoundingBox([mi[0], ma[0]], [mi[1], ma[1]], [mi[2], ma[2]])


def new_curve_object(name, curve):
    return bpy.data.objects.new(name, curve)


def get_curve_for_b_object(bob):
    return bpy.data.curves[bob.ref_obj.name]


def get_curve_for_object(obj):
    return bpy.data.curves[obj.name]


def separate_pieces(data):
    '''
    check whether the data breaks apart into separate pieces
    :param data:
    :return:
    '''

    gaps = [(data[i] - data[i - 1]).length for i in range(1, len(data))]
    gaps.sort()

    # skip the first 20 percent and skip the last 20 percent
    l = len(gaps)
    gaps = gaps[int(0.2 * l):int(0.8 * l)]
    average = sum(gaps) / l

    # split gaps that are larger than 5*average

    pieces = []
    part = [data[0]]
    for i in range(1, len(data)):
        if (data[i] - data[i - 1]).length < 5 * average:
            part.append(data[i])
        else:
            pieces.append(part)
            part = [data[i]]
    pieces.append(part)
    return pieces


def get_new_curve(name, num_points, data=None):
    curve = bpy.data.curves.new(name, type='CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 10
    if data is None:
        add_bezier_spline(curve.splines, num_points, data=data, cyclic=False)
    else:
        data_pieces = separate_pieces(data)
        for data in data_pieces:
            # add first and last again
            data.append(data[-1])
            add_bezier_spline(curve.splines, len(data) - 1, data=data, cyclic=False)
    return curve


def get_splines(obj):
    return obj.data.splines


def get_splines_of_curve(b_obj):
    return get_obj(b_obj).data.spliens


def get_bezier_points(spline):
    return spline.bezier_points


def add_bezier_spline(splines, num_points=2, data=None, cyclic=True):
    new_spline = splines.new('BEZIER')
    new_spline.bezier_points.add(num_points - 1)
    new_spline.use_cyclic_u = cyclic
    if data:
        for bezier, point in zip(new_spline.bezier_points, data):
            bezier.co = Vector([point[0], point[1], 0])
    return new_spline


def set_bezier_point_of_curve(curve, index, position, handle_left, handle_right):
    spline = curve.splines[0]
    if index < len(spline.bezier_points):
        spline.bezier_points[index].co = position
        spline.bezier_points[index].handle_left_type = 'FREE'
        spline.bezier_points[index].handle_left = handle_left
        spline.bezier_points[index].handle_right_type = 'FREE'
        spline.bezier_points[index].handle_right = handle_right


def add_points_to_spline(spline, total_points, closed_loop):
    """
    add bezier points to an existing spline until the total_points number is reached
    The curve of the spline has to be set to edit type before this method can be applied

    Use subdivides to make control points that don't affect shape

    :param spline:
    :param total_points:
    :param closed_loop:
    :return:
    """

    points = spline.bezier_points
    while len(spline.bezier_points) < total_points:
        # find longest segment to subdivide, ignores curvature
        longest = 0
        start_index = 0
        end_index = 1
        for j in range(len(points)):
            if closed_loop:
                k = (j + 1) % len(points)
                sep = points[k].co - points[j].co
                length = sep.length
            else:
                if j == len(points) - 1:
                    k = j
                else:
                    k = j + 1
                length = points[k].co[0] - points[j].co[0]
                # This is a hacky way of making it work for graph curves
                # bpy making it as uniform as possible along x.
                # Doesn't make sense in general.

            if length > longest:
                start_index = j
                end_index = k
                longest = length

        # subdivide longest segments
        points[start_index].select_control_point = True
        points[end_index].select_control_point = True
        # execute_and_time("Get ready to subdivide")
        # execute_and_time(
        #    "Subdivide",
        bpy.ops.curve.subdivide()
        # )
        for point in points:
            point.select_control_point = False


def reset_bezier_point(point, position, handle_left, handle_right):
    point.co = position
    point.handle_left = handle_left
    point.handle_right = handle_right


def set_use_path(bob, is_used):
    curve = get_curve_for_b_object(bob)
    curve.use_path = is_used
    curve.bevel_factor_mapping_end = 'SPLINE'


def set_bevel(bob, depth, caps=True, res=4):
    get_curve_for_b_object(bob).bevel_depth = depth
    get_curve_for_b_object(bob).use_fill_caps = caps
    get_curve_for_b_object(bob).bevel_resolution = res


def change_bevel(bob, old_depth, new_depth, begin_frame=0, transition_frames=DEFAULT_ANIMATION_TIME * FRAME_RATE):
    curve = get_curve_for_b_object(bob)
    curve.bevel_depth = old_depth
    curve.keyframe_insert(data_path="bevel_depth", frame=begin_frame)
    curve.bevel_depth = new_depth
    curve.keyframe_insert(data_path="bevel_depth", frame=begin_frame + transition_frames)


def set_extrude(bob, extrude):
    obj = get_obj(bob)
    data = obj.data
    if hasattr(data, 'extrude'):
        data.extrude = extrude

    # old version before 2022.12.03 (introduction of multilinetexbobject)
    # get_curve_for_b_object(bob).extrude = extrude


def set_bevel_factor_mapping(bob, bevel_factor_mapping_end="RESOLUTION"):
    obj = get_obj(bob)
    data = obj.data
    if hasattr(data, 'bevel_factor_mapping_end'):
        data.bevel_factor_mapping_end = bevel_factor_mapping_end


def grow_curve(bob, appear_frame, duration, inverted=False, start_factor=0, end_factor=1,
               bevel_factor_mapping_end="RESOLUTION"):
    if not hasattr(bob, 'ref_obj'):
        curve = get_curve_for_object(bob)
    else:
        curve = get_curve_for_b_object(bob)
    if hasattr(curve, 'bevel_factor_mapping_end'):
        curve.bevel_factor_mapping_end = bevel_factor_mapping_end

    print("Grow curve: " + curve.name + " at time " + str(appear_frame / FRAME_RATE) + " for " + str(
        duration))
    if not inverted:
        set_bevel_factor_and_keyframe(curve, start_factor, appear_frame)
        set_bevel_factor_and_keyframe(curve, end_factor, appear_frame + duration * FRAME_RATE)
    else:
        set_bevel_factor_start_and_end_keyframe(curve, 1, end_factor, appear_frame)
        set_bevel_factor_start_and_end_keyframe(curve, 1, start_factor, appear_frame + duration * FRAME_RATE)


def set_curve_full_range(bob, factor1, factor2, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
    curve = get_curve_for_object(get_obj(bob))
    appear_frame = begin_time * FRAME_RATE
    frame_duration = np.maximum(1, transition_time * FRAME_RATE)
    set_bevel_factor_start_and_end_keyframe(curve, factor1, factor2, appear_frame + frame_duration)


def set_curve_range(bob, factor, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
    curve = get_curve_for_object(get_obj(bob))
    appear_frame = begin_time * FRAME_RATE
    frame_duration = np.maximum(1, transition_time * FRAME_RATE)
    set_bevel_factor_and_keyframe(curve, factor, appear_frame + frame_duration)


def shrink_curve(bob, appear_frame, duration, inverted=False):
    curve = get_curve_for_object(get_obj(bob))
    print("Grow curve: " + curve.name + " at time " + str(appear_frame / FRAME_RATE) + " for " + str(
        duration / FRAME_RATE))
    if not inverted:
        set_bevel_factor_and_keyframe(curve, 1, appear_frame)
        set_bevel_factor_and_keyframe(curve, 0, appear_frame + duration * FRAME_RATE)
    else:
        set_bevel_factor_start_keyframe(curve, 0, appear_frame)
        set_bevel_factor_start_keyframe(curve, 1, appear_frame + duration * FRAME_RATE)


# all around shape keys
def add_shape_key(bobj, name, previous=None, following=None, relative=True):
    obj = get_obj(bobj)
    # add basis shape key, if it doesn't exist
    if not obj.data.shape_keys and name != 'Basis':  # some functions create the basis sk by themselves (e.g. in curves)
        sk = obj.shape_key_add(name='Basis')
        previous = sk

    # create shapekey for the transformation
    sk = obj.shape_key_add(name=name)
    if previous is not None:
        obj.data.shape_keys.key_blocks[name].relative_key = obj.data.shape_keys.key_blocks[
            previous.name]
    elif following is not None:
        obj.data.shape_keys.key_blocks[following.name].relative_key = obj.data.shape_keys.key_blocks[name]
    sk.interpolation = 'KEY_LINEAR'
    obj.data.shape_keys.use_relative = relative
    return sk


def keyframe_shape_key(bob, sk_name, frame):
    obj = get_obj(bob)
    insert_keyframe(obj.data.shape_keys.key_blocks[sk_name], "value", frame)


def set_shape_key_to_value(bob, sk_name, value, frame):
    obj = get_obj(bob)
    obj.data.shape_keys.key_blocks[sk_name].value = value
    keyframe_shape_key(bob, sk_name, frame)


def set_shape_key_eval_time(bob, time, frame):
    obj = get_obj(bob)
    obj.data.shape_keys.eval_time = time
    obj.data.shape_keys.keyframe_insert(data_path='eval_time', frame=frame)


def create_shape_key_from_transformation(bob, old_sk, index, transformation=lambda x: x):
    obj = get_obj(bob)
    old_sk = add_shape_key(obj, name='tranformation_' + str(index), previous=old_sk)
    for i in range(len(old_sk.data)):
        old_sk.data[i].co = transformation(old_sk.data[i].co)
    return old_sk


def morph_to_next_shape(blender_obj, current_shape_index, appear_frame, frame_duration):
    blender_obj = get_obj(blender_obj)

    next_shape = current_shape_index + 1
    for i in range(next_shape, len(blender_obj.data.shape_keys.key_blocks)):  # set all following blocks to zero
        blender_obj.data.shape_keys.key_blocks[i].value = 0
    insert_keyframe(blender_obj.data.shape_keys.key_blocks[next_shape], "value", appear_frame)

    blender_obj.data.shape_keys.key_blocks[next_shape].value = 1
    insert_keyframe(blender_obj.data.shape_keys.key_blocks[next_shape], "value", appear_frame + frame_duration)


def morph_to_next_shape2(blender_obj, current_shape_index, appear_frame, frame_duration):
    '''
    In this version, always all but the current shape are set to zero
    :param blender_obj:
    :param current_shape_index:
    :param appear_frame:
    :param frame_duration:
    :return:
    '''
    blender_obj = get_obj(blender_obj)

    next_shape = current_shape_index + 1
    for i in range(len(blender_obj.data.shape_keys.key_blocks)):  # set all following blocks to zero
        # set all values at the current value
        value = get_value_at_frame(blender_obj.data.shape_keys.key_blocks[i], appear_frame)
        blender_obj.data.shape_keys.key_blocks[i].value = value
        insert_keyframe(blender_obj.data.shape_keys.key_blocks[i], "value", appear_frame)
        # set all values to zero at the end
        blender_obj.data.shape_keys.key_blocks[i].value = 0.
        insert_keyframe(blender_obj.data.shape_keys.key_blocks[i], "value", appear_frame + frame_duration)

    insert_keyframe(blender_obj.data.shape_keys.key_blocks[next_shape], "value", appear_frame)

    blender_obj.data.shape_keys.key_blocks[next_shape].value = 1
    insert_keyframe(blender_obj.data.shape_keys.key_blocks[next_shape], "value", appear_frame + frame_duration)
    print("Transform to next mesh ", blender_obj.name)


def morph_to_previous_shape(blender_obj, current_shape_index, appear_frame, frame_duration):
    blender_obj = get_obj(blender_obj)
    previous_shape = current_shape_index
    for i in range(previous_shape, len(blender_obj.data.shape_keys.key_blocks)):  # set all following blocks to zero
        blender_obj.data.shape_keys.key_blocks[i].value = 1
    insert_keyframe(blender_obj.data.shape_keys.key_blocks[previous_shape], "value", appear_frame)

    blender_obj.data.shape_keys.key_blocks[previous_shape].value = 0
    insert_keyframe(blender_obj.data.shape_keys.key_blocks[previous_shape], "value", appear_frame + frame_duration)


def set_to_last_shape(blender_obj, appear_frame):
    blender_obj = get_obj(blender_obj)

    for i in range(0, len(blender_obj.data.shape_keys.key_blocks)):  # set all following blocks to zero
        blender_obj.data.shape_keys.key_blocks[i].value = 1
        insert_keyframe(blender_obj.data.shape_keys.key_blocks[i], "value", appear_frame)

    return len(blender_obj.data.shape_keys.key_blocks)


########################
# Work with key frames #
########################

def clear_animation_data(bob):
    ### this only works after the object has been linked
    obj = get_obj(bob)
    if obj.animation_data and obj.animation_data.nla_tracks:
        for nt in obj.animation_data.nla_tracks:
            obj.animation_data.nla_tracks.remove(nt)
    if obj.animation_data and obj.animation_data.drivers:
        for dr in obj.animation_data.drivers:
            obj.animation_data.drivers.remove(dr)

    obj.animation_data_clear()


def start_frame():
    return bpy.data.scenes['Scene'].frame_start


def end_frame():
    return bpy.data.scenes['Scene'].frame_end


def insert_keyframe(bob, data_path, frame):
    obj = get_obj(bob)
    obj.keyframe_insert(data_path=data_path, frame=frame)


def set_bevel_factor_and_keyframe(data, value, frame):
    """
    default case
    set bevel_factor_end, assuming that bevel_factor_start = 0
    :param data:
    :param value:
    :param frame:
    :return:
    """
    set_bevel_factor_start_and_end_keyframe(data, 0, value, int(frame))


def set_bevel_factor_start_and_end_keyframe(data, start, end, frame):
    """
    more flexible,
    start and end value can be controlled independently
    :param data:
    :param start:
    :param end:
    :param frame:
    :return:
    """
    data.bevel_factor_start = start
    data.bevel_factor_end = end
    insert_keyframe(data, 'bevel_factor_end', int(frame))
    insert_keyframe(data, 'bevel_factor_start', int(frame))


def set_bevel_factor_start_keyframe(data, start, frame):
    data.bevel_factor_start = start
    insert_keyframe(data, 'bevel_factor_start', int(frame))


# animations

def set_linear_fcurves(bob):
    obj = get_obj(bob)
    fcurves = obj.animation_data.action.fcurves
    for fcurve in fcurves:
        for kp in fcurve.keyframe_points:
            kp.interpolation = 'LINEAR'


def set_linear_fcurves_for_nodes(node):
    selected_f_curve = None
    for action in bpy.data.actions:
        for fcurve in action.fcurves:
            if node.name in fcurve.data_path:
                selected_f_curve = fcurve
                break
    if selected_f_curve is not None:
        for kp in selected_f_curve.keyframe_points:
            kp.interpolation = 'LINEAR'


def set_linear_action(bob, data_path_part):
    '''
    :param bob: :param data_path_part: part of the data_path such as 'offset_factor', 'location', 'influence' You can
    always access the information about the data path  in bpy.data.actions[obj.name+'Action]. Iterate through all
    fcurves and print fcurve.path_data
    :return:

    '''
    obj = get_obj(bob)
    action = bpy.data.actions[obj.name + 'Action']
    selected_fcurve = None
    for curve in action.fcurves:
        if data_path_part in curve.data_path:
            selected_fcurve = curve

    if selected_fcurve:
        for kp in selected_fcurve.keyframe_points:
            kp.interpolation = 'LINEAR'


def set_linear_action_full(bob):
    obj = get_obj(bob)
    actions = [a for a in bpy.data.actions if obj.name in a.name]
    for a in actions:
        for curve in a.fcurves:
            for kp in curve.keyframe_points:
                kp.interpolation = 'LINEAR'


######################
# work with geometry #
######################


def add_vertex_group(bob, indices=None, name='VertexGroup', weight_function=lambda x: 1):
    obj = get_obj(bob)
    group = obj.vertex_groups.new(name=name)
    if indices is None:
        # add all vertices
        indices = range(0, len(obj.data.vertices))
        for i, v in enumerate(obj.data.vertices):
            loc = v.co
            # print(loc,weight_function(loc))
            group.add([i], weight_function(loc), 'ADD')
    else:
        group.add(indices, 1.0, 'ADD')
    return group


def set_origin(bob, type='ORIGIN_GEOMETRY'):
    """
    :param bob:
    :param type: 'GEOMETRY_ORIGIN', 'ORIGIN_GEOMETRY', 'ORIGIN_CURSOR', 'ORIGIN_CENTER_OF_MASS', 'ORIGIN_CENTER_OF_VOLUME'
    :return:
    """
    obj = get_obj(bob)
    deselect_all()
    select(obj)
    bpy.ops.object.origin_set(type=type)


def set_origin_of_objects_with_name(name=None, type='ORIGIN_GEOMETRY'):
    for o in bpy.data.objects:
        if name in o.name:
            select(o)
            set_origin(o, type="ORIGIN_CENTER_OF_VOLUME")


def set_pivot(obj, origin):
    obj = get_obj(obj)
    if isinstance(origin, list):
        origin = Vector(origin)
    elif isinstance(origin, tuple):
        origin = Vector(origin)
    if obj.parent:  # for the cursor we need the world coordinate of the object, somehow the matrix_world method doesn't work as expected, there nested generation of the matrix_world
        matrix = mathutils.Matrix()
        parent = obj.parent
        while parent:
            matrix = parent.matrix_basis @ matrix
            parent = parent.parent
        set_cursor_location(matrix @ origin)
    else:
        set_cursor_location(origin)
    set_origin(obj, type='ORIGIN_CURSOR')
    cursor_to_origin()


# I/O operations
def save(filename):
    bpy.ops.wm.save_mainfile(filepath=os.path.join(BLEND_DIR, filename + ".blend"))
    print("saved at ", datetime.now(), date.today())


def load(filepath):
    '''
    low level data import
    :param filepath:
    :return:
    '''
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        data_to.objects = data_from.objects
    return data_to.objects


def load_file(filename):
    bpy.ops.wm.load_mainfile(filepath=os.path.join(FINAL_DIR, filename + ".blend"))
    print("loaded at ", datetime.now(), date.today())


# cursor
def set_cursor_location(position):
    bpy.context.scene.cursor.location = position


def cursor_to_origin():
    set_cursor_location([0, 0, 0])


###################
#   modifier      #
###################
def add_sub_division_surface_modifier(b_obj, level=2, adaptive_subdivision=False, dicing_rate=0.5):
    obj = get_obj(b_obj)
    if adaptive_subdivision:
        obj.cycles.use_adaptive_subdivision = True
        obj.cycles.dicing_rate = dicing_rate
    modifier = obj.modifiers.new(name='smooth', type='SUBSURF')
    if modifier:
        modifier.render_levels = level
    return modifier


def add_solidify_modifier(b_obj, thickness=0.1, offset=-1):
    obj = get_obj(b_obj)
    modifier = obj.modifiers.new(name='solidify', type='SOLIDIFY')
    modifier.thickness = thickness
    modifier.offset = offset
    return modifier


def change_solidifier_thickness(bob, from_value=0, to_value=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
    obj = get_obj(bob)
    modifier = obj.modifiers.get('solidify')
    modifier.thickness = from_value
    insert_keyframe(modifier, 'thickness', begin_time * FRAME_RATE)
    modifier.thickness = to_value
    insert_keyframe(modifier, 'thickness', (begin_time + transition_time) * FRAME_RATE)


def add_bevel_modifier(b_obj, width=0.1):
    obj = get_obj(b_obj)
    modifier = obj.modifiers.new(name='bevel', type='BEVEL')
    modifier.width = width
    return modifier


def add_constraint(b_object, type, name=None, **kwargs):
    obj = get_obj(b_object)
    use_legacy_behavior = get_from_kwargs(kwargs, "use_legacy_behavior", False)  # introduced for 4.2 (book)
    if not name:
        name = str(type)
    constraint = obj.constraints.new(type=type)
    if type == 'FOLLOW_PATH':
        target = get_from_kwargs(kwargs, 'target', None)
        use_fixed_location = get_from_kwargs(kwargs, 'use_fixed_location', True)
        use_curve_follow = get_from_kwargs(kwargs, 'use_curve_follow', True)
        constraint.target = get_obj(target)
        constraint.use_fixed_location = use_fixed_location
        constraint.use_curve_follow = use_curve_follow
    elif type == 'PIVOT':
        target = get_from_kwargs(kwargs, 'target', None)
        constraint.target = get_obj(target)
        rotation_range = get_from_kwargs(kwargs, 'rotation_range', 'ALWAYS_ACTIVE')
        constraint.rotation_range = rotation_range
    elif type == 'COPY_ROTATION':
        target = get_from_kwargs(kwargs, 'target', None)
        constraint.target = get_obj(target)
        mix_mode = get_from_kwargs(kwargs, 'mix_mode', 'REPLACE')
        constraint.mix_mode = mix_mode
    elif type == 'LIMIT_ROTATION':
        limit_x = get_from_kwargs(kwargs, 'limit_x', False)
        limit_y = get_from_kwargs(kwargs, 'limit_y', False)
        limit_z = get_from_kwargs(kwargs, 'limit_z', False)
        min_x = get_from_kwargs(kwargs, 'min_x', 0)
        max_x = get_from_kwargs(kwargs, 'max_x', 0)
        min_y = get_from_kwargs(kwargs, 'min_y', 0)
        max_y = get_from_kwargs(kwargs, 'max_y', 0)
        min_z = get_from_kwargs(kwargs, 'min_z', 0)
        max_z = get_from_kwargs(kwargs, 'max_z', 0)
        constraint.use_limit_x = limit_x
        constraint.use_limit_y = limit_y
        constraint.use_limit_z = limit_z
        constraint.min_x = min_x
        constraint.min_y = min_y
        constraint.min_z = min_z
        constraint.max_x = max_x
        constraint.max_y = max_y
        constraint.max_z = max_z
        constraint.use_legacy_behavior = use_legacy_behavior
    elif type == 'COPY_LOCATION':
        target = get_from_kwargs(kwargs, 'target', None)
        constraint.target = get_obj(target)
    return constraint


def change_modifier_attribute(bob, mod_name, attr_name, start, end, begin_frame=0, frame_duration=1):
    obj = get_obj(bob)
    if obj is not None:
        modifier = obj.modifiers[mod_name]
        if modifier is not None:
            setattr(modifier, attr_name, start)
            insert_keyframe(modifier, attr_name, begin_frame)
            setattr(modifier, attr_name, end)
            insert_keyframe(modifier, attr_name, begin_frame + frame_duration)


def add_modifier(b_obj, type, name=None):
    if name is None:
        name = 'mod_' + str(type)
    return add_modifier_recursive(get_obj(b_obj), type, name=name)


def add_modifier_recursive(obj, type, name=None):
    if not name:
        name = str(type)
    if obj.data:
        if obj.type == 'CURVE':
            convert_to_mesh(obj)
        else:
            modifier = obj.modifiers.new(name=name, type=type)
        if type == 'PARTICLE_SYSTEM':
            return modifier.particle_system.settings
        else:
            return modifier
    else:
        modifiers = []
        for child in obj.children:
            modifiers.append(add_modifier_recursive(child, type))
        return modifiers


def add_mesh_modifier(bob, **kwargs):
    '''
    This is a new start for a more generic modifier function
    :param bob:
    :param kwargs:
    :return:
    '''
    obj = get_obj(bob)
    type = kwargs.pop('type')
    if 'name' in kwargs:
        name = kwargs.pop('name')
    else:
        name = 'modifier_' + str(type)
    modifiers = obj.modifiers.new(type=type, name=name)
    if type == 'SOLIDIFY':
        if 'thickness' in kwargs:
            thickness = kwargs.pop('thickness')
            modifiers.thickness = thickness
        if 'offset' in kwargs:
            offset = kwargs.pop('offset')
            modifiers.offset = offset
    if type == 'BEVEL':
        if 'amount' in kwargs:
            amount = kwargs.pop('amount')
            modifiers.width = amount
        if 'segments' in kwargs:
            segments = kwargs.pop('segments')
            modifiers.segments = segments
    if type == 'SUBSURF':
        levels = get_from_kwargs(kwargs, "levels", 1)
        render_levels = get_from_kwargs(kwargs, "render_levels", 2)
        modifiers.levels = levels
        modifiers.render_levels = render_levels
    if type == 'WIREFRAME':
        if 'thickness' in kwargs:
            thickness = kwargs.pop('thickness')
            modifiers.thickness = thickness
        if 'use_even_offset' in kwargs:
            use_even_offset = kwargs.pop('use_even_offset')
            modifiers.use_even_offset = use_even_offset
    elif type == 'ARRAY':
        if 'count' in kwargs:
            count = kwargs.pop('count')
            modifiers.count = count
        if 'relative_offset_displace' in kwargs:
            relative_offset_displace = kwargs.pop('relative_offset_displace')
            modifiers.relative_offset_displace = relative_offset_displace
    elif type == 'SIMPLE_DEFORM':
        deform_method = get_from_kwargs(kwargs, 'deform_method', 'TWIST')
        deform_axis = get_from_kwargs(kwargs, 'deform_axis', 'X')
        angle = get_from_kwargs(kwargs, 'angle', 0)
        modifiers.deform_method = deform_method
        modifiers.deform_axis = deform_axis
        modifiers.angle = angle
    elif type == 'HOOK':
        object = get_from_kwargs(kwargs, 'object', None)
        vertex_group = get_from_kwargs(kwargs, 'vertex_group', None)
        modifiers.object = get_obj(object)
        if vertex_group is not None:
            modifiers.vertex_group = vertex_group.name
    elif type == 'NODES':
        if 'node_modifier' in kwargs:
            node_modifier = kwargs.pop('node_modifier')
            modifiers.node_group = node_modifier.get_node_tree()
            if 'attribute_names' in kwargs:
                attribute_names = kwargs.pop('attribute_names')
                for i, name in enumerate(attribute_names):
                    modifiers["Output_" + str(i + 1) + "_attribute_name"] = name
            # transfer possible color to the material slot of the blender object
            for mat in node_modifier.materials:
                obj.data.materials.append(mat)
        if 'node_group' in kwargs:
            node_group = kwargs.pop('node_group')
            modifiers.node_group = node_group
            if 'attribute_names' in kwargs:
                attribute_names = kwargs.pop('attribute_names')
                for i, name in enumerate(attribute_names):
                    modifiers["Output_" + str(i + 1) + "_attribute_name"] = name


def apply_modifiers(bob):
    obj = get_obj(bob)
    select(obj)
    set_active(obj)
    while len(obj.modifiers) > 0:
        bpy.ops.object.modifier_apply(modifier=obj.modifiers[0].name)


def apply_modifier(bob, type=None):
    obj = get_obj(bob)
    if type is not None:
        if type in obj.modifiers:
            select(obj)
            set_active(obj)
            bpy.ops.object.modifier_apply(modifier=obj.modifiers.get(type).name)


####################
# Transformations  #
####################

def set_taa_render_samples(taa_render_samples=64, begin_frame=0):
    scene = get_scene()
    if scene.render.engine == BLENDER_EEVEE:
        set_frame(begin_frame - 1)
        old_taa_render_samples = scene.eevee.taa_render_samples
        scene.eevee.taa_render_samples = old_taa_render_samples
        scene.keyframe_insert(data_path="eevee.taa_render_samples", frame=begin_frame - 1)
        scene.eevee.taa_render_samples = taa_render_samples
        scene.keyframe_insert(data_path="eevee.taa_render_samples", frame=begin_frame)


def disappear_all_copies_of_letters(begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
    objects = bpy.data.objects
    for obj in objects:
        if 'copy_of_ren' in obj.name:
            recursive_fade_out(obj, begin_time * FRAME_RATE, transition_time * FRAME_RATE)


def fade_in(b_obj, frame, frame_duration, **kwargs):
    alpha = get_from_kwargs(kwargs, 'alpha', 1)
    offset_for_slots = get_from_kwargs(kwargs, 'offset_for_slots', None)

    obj = get_obj(b_obj)

    set_alpha_and_keyframe(obj, 0, int(frame), offset_for_slots=offset_for_slots)
    set_alpha_and_keyframe(obj, alpha, int(frame + frame_duration), offset_for_slots=offset_for_slots)

    if frame_duration == 1:
        unhide_frm(b_obj, frame + frame_duration)
    else:
        unhide_frm(b_obj, frame)
    # recursively_fade_in(obj,alpha,frame,frame_duration)


def recursively_fade_in(obj, alpha, frame, frame_duration):
    set_alpha_and_keyframe(obj, 0, frame)
    set_alpha_and_keyframe(obj, alpha, frame + frame_duration)

    for child in obj.children:
        print("DEPRECATED: fade in recursively " + child.name)
        recursively_fade_in(child, alpha, frame, frame_duration)


def change_alpha_of_material(mat, from_value=0, to_value=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
    nodes = mat.node_tree.nodes
    if "Principled BSDF" in nodes:
        return change_default_value(nodes["Principled BSDF"].inputs['Alpha'], from_value=from_value, to_value=to_value,
                                    begin_time=begin_time, transition_time=transition_time)
    else:
        raise "Cannot change Alpha for non BSDF nodes"


def change_alpha(b_obj, frame, frame_duration, alpha=0):
    """
    fade out b_object and hide it from scene
    it is recursively applied to all the children

    :param handwriting: special for text
    :param alpha: degree of disappearance
    :param b_obj:
    :param frame:
    :param frame_duration: 
    :return:
    """
    obj = get_obj(b_obj)
    alpha0 = get_alpha_at_current_keyframe(obj, frame)
    set_alpha_and_keyframe(obj, alpha0, frame)
    set_alpha_and_keyframe(obj, alpha, frame + frame_duration)


def change_shader_value(b_object, node, input, initial_value=0, final_value=1, frame=0,
                        frame_duration=DEFAULT_ANIMATION_TIME * FRAME_RATE):
    obj = get_obj(b_object)
    material = get_material_of(obj)
    nodes = material.node_tree.nodes
    node = nodes[node]
    node.inputs[input].default_value = initial_value
    insert_keyframe(node.inputs[input], 'default_value', frame)
    node.inputs[input].default_value = final_value
    insert_keyframe(node.inputs[input], 'default_value', frame + frame_duration)


def fade_out(b_obj, frame, frame_duration, alpha=0, handwriting=False, **kwargs):
    """
    fade out b_object and hide it from scene
    it is recursively applied to all the children
    
    :param handwriting: special for text
    :param alpha: degree of disappearance
    :param b_obj:
    :param frame: 
    :param frame_duration: 
    :return: 
    """
    obj = get_obj(b_obj)

    recursive_fade_out(obj, frame, frame_duration, handwriting=handwriting, alpha=alpha, **kwargs)

    if alpha == 0:
        hide_frm(b_obj, frame + frame_duration)


def fade_out_quickly(obj, frame, frame_duration):
    """
    quick fade out assumes that the object was fully visible at the time of disappearance
    :param obj:
    :param frame:
    :param frame_duration:
    :return:
    """
    set_alpha_and_keyframe(obj, 1, frame)
    set_alpha_and_keyframe(obj, 0, frame + frame_duration)


def recursive_fade_out(obj, frame, frame_duration, handwriting=False, alpha=0, slot=0):
    # retrieve current alpha state  to fade out from the current state
    if not "hand_written" in obj.name or handwriting:
        alpha0 = get_alpha_at_current_keyframe(obj, frame, slot)
        set_alpha_and_keyframe(obj, alpha0, frame)
        set_alpha_and_keyframe(obj, alpha, frame + frame_duration)

    for child in obj.children:
        recursive_fade_out(child, frame, frame_duration, alpha=alpha)

    if alpha == 0:
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_render", frame=frame + frame_duration)


def move(b_obj, direction, begin_frame, frame_duration):
    """
    move b_object for a given distance
    
    :param total_motion:
    :param b_obj:
    :param direction: 
    :param begin_frame: 
    :param frame_duration: 
    :return: 
    """
    obj = get_obj(b_obj)
    obj.location = get_location_at_frame(b_obj, np.maximum(0, begin_frame - 1))
    insert_keyframe(obj, "location", int(begin_frame))
    # print("for "+obj.name+" location "+str(obj.location)+" keyframed at frame "+str(int(begin_frame+1)))

    obj.location += direction
    insert_keyframe(obj, "location", int(begin_frame + np.maximum(1, frame_duration)))
    # print("for " + obj.name + " location " + str(obj.location) + " keyframed at frame " + str(int(begin_frame+frame_duration)))


def move_fast_from_to(b_obj, start=Vector(), end=Vector(), begin_frame=0,
                      frame_duration=DEFAULT_ANIMATION_TIME * FRAME_RATE):
    obj = get_obj(b_obj)
    obj.location = start
    insert_keyframe(obj, "location", begin_frame)
    obj.location = end
    insert_keyframe(obj, "location", begin_frame + frame_duration)


def move_to(b_obj, target, begin_frame, frame_duration, global_system=False):
    location = get_location_at_frame(b_obj, begin_frame - 1)
    obj = get_obj(b_obj)
    obj.location = location
    insert_keyframe(obj, "location", begin_frame)
    if global_system:
        target = obj.parent.matrix_world.inverted() @ target
    obj.location = target
    insert_keyframe(obj, "location", begin_frame + frame_duration)
    print(
        "MoveTo " + obj.name + " at time " + str(begin_frame / FRAME_RATE) + " for " + str(
            frame_duration / FRAME_RATE) + " seconds.")


def rotate_by(b_obj, rotation_euler):
    obj = get_obj(b_obj)
    old_euler = obj.rotation_euler.copy()
    new_euler = (mathutils.Euler(rotation_euler).to_matrix() @ mathutils.Euler(old_euler).to_matrix()).to_euler()
    obj.rotation_euler = new_euler


def rotate_to(b_obj, begin_frame, frame_duration, pivot, interpolation, **kwargs):
    obj = get_obj(b_obj)
    frame_duration = np.maximum(1, frame_duration)
    if pivot is not None:
        set_pivot(obj, pivot)
    if 'rotation_euler' in kwargs:
        rotation_euler = get_rotation_at_frame(b_obj, begin_frame - 1)
        obj.rotation_euler = rotation_euler
        insert_keyframe(obj, "rotation_euler", begin_frame)
        rotation_euler = kwargs['rotation_euler']
        obj.rotation_mode = 'XYZ'
        obj.rotation_euler = rotation_euler
        insert_keyframe(obj, "rotation_euler", begin_frame + frame_duration)
    elif 'rotation_quaternion' in kwargs:
        rotation_quat = get_rotation_quaternion_at_frame(b_obj, begin_frame - 1)
        obj.rotation_quaternion = rotation_quat
        insert_keyframe(obj, "rotation_quaternion", begin_frame)
        rotation_quaternion = kwargs['rotation_quaternion']
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = rotation_quaternion
        insert_keyframe(obj, "rotation_quaternion", begin_frame + frame_duration)

    # print(
    #     "Rotate " + obj.name + " at time " + str(begin_frame / FRAME_RATE) + " for " + str(
    #         frame_duration / FRAME_RATE) + " seconds.")

    if interpolation != 'BEZIER':
        f_curves = obj.animation_data.action.fcurves
        for curve in f_curves:
            for kf in curve.keyframe_points:
                kf.interpolation = interpolation


def shrink(b_obj, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
    obj = get_obj(b_obj)
    frame = begin_time * FRAME_RATE
    duration = transition_time * FRAME_RATE
    obj.scale = get_scale_at_frame(b_obj, frame - 1)
    insert_keyframe(obj, "scale", frame - 1)
    obj.scale = [0, 0, 0]
    insert_keyframe(obj, "scale", frame + duration)
    # for c in obj.children:
    #     shrink(c,begin_time=begin_time,transition_time=transition_time)


def grow_from(b_obj, pivot, begin_frame, frame_duration):
    obj = get_obj(b_obj)
    if pivot:
        set_pivot(obj, pivot)

    scale = obj.scale.copy()
    obj.scale = [0] * 3
    insert_keyframe(obj, "scale", int(begin_frame))

    obj.scale = scale
    insert_keyframe(obj, "scale", int(begin_frame + np.maximum(1, frame_duration)))


def rescale(b_obj, re_scale, begin_frame, frame_duration):
    obj = get_obj(b_obj)

    if not isinstance(re_scale, list):
        re_scale = [re_scale] * 3

    scale = get_scale_at_frame(b_obj, begin_frame)
    scale_new = [scale[0] * re_scale[0], scale[1] * re_scale[1], scale[2] * re_scale[2]]
    obj.scale = scale
    insert_keyframe(obj, "scale", begin_frame)
    obj.scale = scale_new
    insert_keyframe(obj, "scale", begin_frame + frame_duration)


def grow(b_obj, scale, begin_frame, frame_duration, initial_scale=0, modus='from_center'):
    obj = get_obj(b_obj)
    select(obj)
    if modus == 'from_center':
        pivot = None
    elif modus == 'from_left':
        # move center of geometry to the smallest x-value
        pivot = find_center_of_leftest_vertices(obj)
    elif modus == 'from_right':
        # move center of geometry to the largest x-value
        pivot = find_center_of_rightest_vertices(obj)
    elif modus == 'from_bottom' or modus == 'from_start':
        # move center of geometry to the smallest x-value
        pivot = find_center_of_lowest_vertices(obj)
    elif modus == 'from_top':
        # move center of geometry to the smallest x-value
        pivot = find_center_of_highest_vertices(obj)
    elif modus == 'from_front':
        # move center of geometry to the smallest x-value
        pivot = find_center_of_closed_vertices(obj)
    elif modus == 'from_back':
        # move center of geometry to the smallest x-value
        pivot = find_center_of_furthest_vertices(obj)

    print(
        "Grow " + obj.name + " at time " + str(begin_frame / FRAME_RATE) + " for " + str(
            frame_duration / FRAME_RATE) + " seconds.")

    if pivot is not None:
        set_pivot(obj, pivot)

    if isinstance(initial_scale, Vector):
        obj.scale = [initial_scale[0], initial_scale[1], initial_scale[2]]
    elif isinstance(initial_scale, list):
        obj.scale = initial_scale
    else:
        obj.scale = [initial_scale] * 3

    insert_keyframe(obj, "scale", begin_frame)
    if isinstance(scale, Vector):
        obj.scale = [scale[0], scale[1], scale[2]]
    elif isinstance(scale, list):
        obj.scale = scale
    else:
        obj.scale = [scale] * 3

    insert_keyframe(obj, "scale", begin_frame + np.maximum(1, frame_duration))


#######################
# Auxiliary functions #
#######################

def find_center_of_leftest_vertices(obj):
    """
    find the geometrical center of the left most face, edge or vertex
    :param obj:
    :return:
    """
    vertices = obj.data.vertices
    m = obj.matrix_world

    x_min = float("inf")
    # convert vertex coordinates into world coordinates
    world_coords = []
    for vertex in vertices:
        v4 = Vector([*vertex.co, 1])
        v = m @ v4
        v_real = Vector([v.x / v.w, v.y / v.w, v.z / v.w])
        world_coords.append(v_real)
        if v_real.x < x_min:
            x_min = v_real.x

    count = 0
    y_left = 0
    z_left = 0
    for wc in world_coords:
        if close(wc.x, x_min):
            count += 1
            y_left += wc.y
            z_left += wc.z

    y_left /= count
    z_left /= count

    return x_min, y_left, z_left


def find_center_of_rightest_vertices(obj):
    """
    find the geometrical center of the left most face, edge or vertex
    :param obj:
    :return:
    """

    vertices = obj.data.vertices
    m = obj.matrix_world

    x_max = -float("inf")
    # convert vertex coordinates into world coordinates
    world_coords = []
    for vertex in vertices:
        v4 = Vector([*vertex.co, 1])
        v = m @ v4
        v_real = Vector([v.x / v.w, v.y / v.w, v.z / v.w])
        world_coords.append(v)
        if v_real.x > x_max:
            x_max = v_real.x

    count = 0
    y_left = 0
    z_left = 0
    for wc in world_coords:
        if close(wc.x, x_max):
            count += 1
            y_left += wc.y
            z_left += wc.z

    y_left /= count
    z_left /= count

    return x_max, y_left, z_left


def find_center_of_closed_vertices(obj):
    """
    find the geometrical center of the left most face, edge or vertex
    :param obj:
    :return:
    """
    vertices = obj.data.vertices
    m = obj.matrix_world

    y_min = float("inf")
    # convert vertex coordinates into world coordinates
    world_coords = []
    for vertex in vertices:
        v4 = Vector([*vertex.co, 1])
        v = m @ v4
        v_real = Vector([v.x / v.w, v.y / v.w, v.z / v.w])
        world_coords.append(v_real)
        if v_real.y < y_min:
            y_min = v_real.y

    count = 0
    x_front = 0
    z_front = 0
    for wc in world_coords:
        if close(wc.y, y_min):
            count += 1
            x_front += wc.x
            z_front += wc.z

    x_front /= count
    z_front /= count

    return x_front, y_min, z_front


def find_center_of_furthest_vertices(obj):
    """
    find the geometrical center of the left most face, edge or vertex
    :param obj:
    :return:
    """

    vertices = obj.data.vertices
    m = obj.matrix_world

    y_max = -float("inf")
    # convert vertex coordinates into world coordinates
    world_coords = []
    for vertex in vertices:
        v4 = Vector([*vertex.co, 1])
        v = m @ v4
        v_real = Vector([v.x / v.w, v.y / v.w, v.z / v.w])
        world_coords.append(v)
        if v_real.y > y_max:
            y_max = v_real.y

    count = 0
    x_back = 0
    z_back = 0
    for wc in world_coords:
        if close(wc.y, y_max):
            count += 1
            x_back += wc.x
            z_back += wc.z

    x_back /= count
    z_back /= count

    return x_back, y_max, z_back


def find_center_of_lowest_vertices(obj):
    """
    find the geometrical center of the left most face, edge or vertex
    :param: obj
    :return:
    """

    vertices = obj.data.vertices
    m = obj.matrix_world

    z_min = float("inf")
    # convert vertex coordinates into world coordinates
    world_coords = []
    for vertex in vertices:
        v4 = Vector([*vertex.co, 1])
        v = m @ v4
        v_real = Vector([v.x / v.w, v.y / v.w, v.z / v.w])
        world_coords.append(v)
        if v_real.z < z_min:
            z_min = v_real.z

    count = 0
    x_low = 0
    y_low = 0
    for wc in world_coords:
        if close(wc.z, z_min):
            count += 1
            x_low += wc.x
            y_low += wc.y

    x_low /= count
    y_low /= count

    return x_low, y_low, z_min


def find_center_of_highest_vertices(obj):
    """
    find the geometrical center of the highest face, edge or vertex
    :param: obj
    :return:
    """

    vertices = obj.data.vertices
    m = obj.matrix_world

    z_max = -float("inf")
    # convert vertex coordinates into world coordinates
    world_coords = []
    for vertex in vertices:
        v4 = Vector([*vertex.co, 1])
        v = m @ v4
        v_real = Vector([v.x / v.w, v.y / v.w, v.z / v.w])
        world_coords.append(v)
        if v_real.z > z_max:
            z_max = v_real.z

    count = 0
    x_high = 0
    y_high = 0
    for wc in world_coords:
        if close(wc.z, z_max):
            count += 1
            x_high += wc.x
            y_high += wc.y

    if count > 0:
        x_high /= count
        y_high /= count

    return x_high, y_high, z_max


'''
Getter and setter for various parameters
'''


def get_bounding_box(bob):
    o = get_obj(bob)
    x_min = np.Infinity
    x_max = -np.Infinity
    y_min = np.Infinity
    y_max = -np.Infinity
    z_min = np.Infinity
    z_max = -np.Infinity
    bb = o.bound_box  # eight corner coordinates of the surrounding box
    for b in bb:
        if b[0] < x_min:
            x_min = b[0]
        if b[0] > x_max:
            x_max = b[0]
        if b[1] < y_min:
            y_min = b[1]
        if b[1] > y_max:
            y_max = b[1]
        if b[2] < z_min:
            z_min = b[2]
        if b[2] > z_max:
            z_max = b[2]
    return [x_min, y_min, z_min, x_max, y_max, z_max]


def get_bounding_box_for_letter(letter):
    x_min = np.Infinity
    x_max = -np.Infinity
    y_min = np.Infinity
    y_max = -np.Infinity
    z_min = np.Infinity
    z_max = -np.Infinity
    obj = letter.ref_obj
    loc = obj.location
    bb = obj.bound_box  # eight corner coordinates of the surrounding box
    for b in bb:
        if (b[0] + loc[0]) < x_min:
            x_min = b[0] + loc[0]
        if (b[0] + loc[0]) > x_max:
            x_max = b[0] + loc[0]
        if (b[1] + loc[1]) < y_min:
            y_min = b[1] + loc[1]
        if (b[1] + loc[1]) > y_max:
            y_max = b[1] + loc[1]
        if (b[2] + loc[2]) < z_min:
            z_min = b[2] + loc[2]
        if (b[2] + loc[2]) > z_max:
            z_max = b[2] + loc[2]
    return [x_min, y_min, z_min, x_max, y_max, z_max]


def analyse_bound_box(bob):
    obj = get_obj(bob)
    bb = obj.bound_box
    xmin = ymin = zmin = np.Infinity
    xmax = ymax = zmax = -np.Infinity
    for b in bb:
        if b[0] < xmin:
            xmin = b[0]
        if b[0] > xmax:
            xmax = b[0]
        if b[1] < ymin:
            ymin = b[1]
        if b[1] > ymax:
            ymax = b[1]
        if b[2] < zmin:
            zmin = b[2]
        if b[2] > zmax:
            zmax = b[2]
    if xmin == 0 and xmax == 0 and ymin == 0 and ymax == 0 and zmin == 0 and zmax == 0:
        xmin = ymin = zmin = np.Infinity
        xmax = ymax = zmax = -np.Infinity
        for child in obj.children:
            [cxmin, cxmax, cymin, cymax, czmin, czmax] = analyse_bound_box(child)
            location = child.location
            cxmin += location[0]
            cxmax += location[0]
            cymin += location[1]
            cymax += location[1]
            czmin += location[2]
            czmax += location[2]
            if cxmin < xmin:
                xmin = cxmin
            if cxmax > xmax:
                xmax = cxmax
            if cymin < ymin:
                ymin = cymin
            if cymax > ymax:
                ymax = cymax
            if czmin < zmin:
                zmin = czmin
            if czmax > zmax:
                zmax = czmax
        return [xmin, xmax, ymin, ymax, zmin, zmax]
    else:
        location = obj.location
        return [xmin + location[0], xmax + location[0], ymin + location[1], ymax + location[1], zmin + location[2],
                zmax + location[2]]


def get_dimension(bob, direction):
    minmax = analyse_bound_box(bob)
    return minmax[2 * direction + 1] - minmax[2 * direction]


def get_world_location(b_obj):
    obj = get_obj(b_obj)
    parent = get_oldest_parent(obj)
    matrix = parent.matrix_world
    return matrix @ obj.location


def get_world_matrix(bob):
    return bob.ref_obj.matrix_world


def get_location(b_obj):
    obj = get_obj(b_obj)
    location = obj.location.copy()
    # if location.dot(location)==0:
    #     # possibility that the object is following a curve
    #     location = obj.matrix_world.translation
    return location


def get_scale(b_obj):
    obj = get_obj(b_obj)
    return obj.scale.copy()


def get_world_location_at_time(b_obj, time):
    frame_old = get_frame()
    frame = time * FRAME_RATE
    set_frame(frame)
    location = get_world_location(b_obj)
    set_frame(frame_old)
    return location


def get_offset_factor_at_frame(b_obj, target, frame):
    constraint = FOLLOW_PATH_DICTIONARY[(b_obj, target)]
    set_frame(frame)
    return constraint.offset_factor


def get_location_at_frame(b_obj, frame):
    frame_old = get_frame()
    set_frame(int(frame))
    location = get_location(b_obj)
    set_frame(frame_old)
    return location


def get_scale_at_frame(b_obj, frame):
    frame_old = get_frame()
    set_frame(frame)
    scale = get_scale(b_obj)
    set_frame(frame_old)
    return scale


def get_rotation_at_frame(b_obj, frame):
    obj = get_obj(b_obj)
    frame_old = get_frame()
    set_frame(frame)
    rotation = obj.rotation_euler.copy()
    set_frame(frame_old)
    return rotation


def get_rotation_quaternion_at_frame(b_obj, frame):
    frame_old = get_frame()
    set_frame(frame)
    rotation = get_obj(b_obj).rotation_quaternion.copy()
    set_frame(frame_old)
    return rotation


def get_input_value_at_frame(b_obj, frame):
    """
    get value of a simple input node
    :param b_obj:
    :param frame:
    :return:
    """
    frame_old = get_frame()
    set_frame(frame)
    value = get_obj(b_obj).outputs[0].default_value
    set_frame(frame_old)
    return value


def transform_into_world(b_obj, location):
    """
    transform a vector into the world coordinates
    :param b_obj:
    :param location:
    :return:
    """
    obj = get_obj(b_obj)
    parent = get_oldest_parent(obj)
    matrix = parent.matrix_world
    return matrix @ location


def set_location(b_obj, location):
    get_obj(b_obj).location = location


##############
## physics ###
##############

def add_rigid_body(bob, **kwargs):
    obj = get_obj(bob)
    select(obj)
    bpy.ops.rigidbody.objects_add(**kwargs)


def key_frame_rigid_body_properties(bob, type='dynamic', value=True, begin_time=0):
    """
    set keyframes for rigid_body properties
    :param bob:
    :param type: 'dynamic', 'kinematic'
    :param value:
    :param begin_time:
    :return:
    """
    obj = get_obj(bob)
    if type == 'dynamic':
        obj.rigid_body.enabled = value
        insert_keyframe(obj.rigid_body, data_path='enabled', frame=begin_time * FRAME_RATE)
    elif type == 'kinematic':
        obj.rigid_body.kinematic = value
        insert_keyframe(obj.rigid_body, data_path='kinematic', frame=begin_time * FRAME_RATE)


def make_rigid_body(bob, dynamic=True, kinematic=False, friction=0.1, bounciness=0.5,
                    all_similar_objects=False, use_margin=False, collision_margin=0.04, **kwargs):
    obj = get_obj(bob)
    set_active(obj)
    set_select(obj, value=True)
    # bpy.ops.rigidbody.world_add()
    similar_objects = []
    if all_similar_objects:
        for o in bpy.data.objects:
            if remove_digits(obj.name) in o.name:
                similar_objects.append(o)
        set_several_select(similar_objects, True)
    bpy.ops.rigidbody.objects_add(**kwargs)
    if obj.rigid_body is not None:
        obj.rigid_body.enabled = dynamic
        obj.rigid_body.kinematic = kinematic
        obj.rigid_body.use_margin = use_margin
        obj.rigid_body.restitution = bounciness
        obj.rigid_body.friction = friction
        obj.rigid_body.collision_margin = collision_margin

    if len(similar_objects) > 0:
        bpy.ops.rigidbody.object_settings_copy()
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")


def set_simulation(begin_time=0, transition_time=250 / FRAME_RATE):
    scene = bpy.data.scenes["Scene"]
    scene.use_custom_simulation_range = True
    start = int(begin_time * FRAME_RATE)
    scene.simulation_frame_start = start
    end = int(begin_time + transition_time) * FRAME_RATE
    scene.simulation_frame_end = end
    scene.rigidbody_world.point_cache.frame_start = start
    scene.rigidbody_world.point_cache.frame_end = end

    return begin_time + transition_time


##################
#   parenting    #
##################

def clear_parent(bob):
    obj = get_obj(bob)
    obj.parent = None


def get_oldest_parent(obj):
    if obj.parent:
        return get_oldest_parent(obj.parent)
    else:
        return obj


def set_parent(b_obj, parent):
    obj = get_obj(b_obj)
    parent = get_obj(parent)
    obj.parent = parent


def set_vertex_parent(parent, child, vertex_index):
    """
    this is used for the chain of arrows
    :param parent:
    :param child:
    :param vertex_index:
    :return:
    """
    parent = get_obj(parent)
    child = get_obj(child)
    child.parent = parent
    child.parent_type = 'VERTEX'
    child.parent_vertices = [vertex_index] * 3


def copy(b_obj):
    obj = get_obj(b_obj)
    copy = obj.copy()
    if copy.data:
        copy.data = copy.data.copy()
    if copy.animation_data and copy.animation_data.action is not None:
        copy.animation_data.action = copy.animation_data.action.copy()
    return copy


def duplicate(b_obj):
    obj = get_obj(b_obj)
    set_active(obj)
    select_recursively(obj)

    bpy.ops.object.duplicate()
    copy = bpy.context.active_object

    # copy.scale = obj.scale
    # copy.rotation_euler = obj.rotation_euler.copy()
    copy.location = obj.location.copy()

    # add all children to copy
    # for o in bpy.context.scene.objects:
    #     if o.select_get() and o != copy:
    #         o.parent = copy

    return copy


#################
# after effects #
#################

def set_exposure(value):
    scene = bpy.context.scene
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links
    exposure = nodes.new(type='CompositorNodeExposure')
    exposure.inputs['Exposure'].default_value = value
    composite = nodes['Composite']
    layers = nodes['Render Layers']
    links.new(layers.outputs['Image'], exposure.inputs['Image'])
    links.new(exposure.outputs['Image'], composite.inputs['Image'])


########################
# actions  and fcurves #
########################
def value_of_action_fcurve_at_frame(action, fcurve, frm):
    '''
    find the value of an f_curve at a particular frame

    first used in video_apollonian

    :param action:
    :param fcurve:
    :param frm:
    :return:
    '''
    return bpy.data.actions[action].fcurves[fcurve].evaluate(frm)


def print_actions():
    '''
    for debugging purposes to find the correct action
    first used in video_apollonian
    '''
    for action in bpy.data.actions:
        print(action)


def print_fcurves_of_action(action):
    '''
    for debugging purposes to find the currect fcurve
     first used in video_apollonian
    :param action:
    :return:
    '''
    for fcurve in bpy.data.actions[action].fcurves:
        print(fcurve)


#############
# attributes#
#############

def add_attribute(bob, name="attribute", attribute=None, type='FLOAT', domain='POINT'):
    obj = get_obj(bob)
    if attribute:
        attr = obj.data.attributes.new(name=name, type=type, domain=domain)
        attr.data.foreach_set('value', attribute)


#############
# utilities #
#############


def vectorize(list):
    return Vector(list)


def diff(a, b):
    return a - b


####################
# append resources #
####################

def append(filename):
    path = os.path.join(APPEND_DIR, filename)

    import_objects = ['Mountains', 'Lake', 'BackgroundMountains']
    import_objects += ['ast_spruce01b', 'ast_spruce01', 'ast_rocklp11', 'ast_rocklp10', 'ast_rocklp09', 'ast_rocklp08',
                       'ast_rocklp07', 'ast_rocklp06', 'ast_rocklp05', 'ast_rocklp04', 'ast_rocklp03', 'ast_rocklp01',
                       'ast_pine02b', 'ast_pine02', 'ast_pine01b', 'ast_pine01', 'ast_ocklp02', 'ast_deciduous03',
                       'ast_deciduous02', 'ast_deciduous01']
    # link all objects in the list
    with bpy.data.libraries.load(path, link=True) as (data_from, data_to):
        data_to.objects = [name for name in data_from.objects if name in import_objects]

    # link object to scene collection
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)

    # with bpy.data.libraries.load(path) as (data_from, data_to):
    #     data_to.collections = [collection for collection in data_from.collections]
    #     print("Created collections: ",data_to.collections)
    #
    #     objects  = ['Mountains', 'Lake', 'Empty', 'Camera', 'BackgroundMountains', 'ast_spruce01b', 'ast_spruce01', 'ast_rocklp11', 'ast_rocklp10', 'ast_rocklp09', 'ast_rocklp08', 'ast_rocklp07', 'ast_rocklp06', 'ast_rocklp05', 'ast_rocklp04', 'ast_rocklp03', 'ast_rocklp01', 'ast_pine02b', 'ast_pine02', 'ast_pine01b', 'ast_pine01', 'ast_ocklp02', 'ast_deciduous03', 'ast_deciduous02', 'ast_deciduous01']
    #
    #     data_to.objects = [name for name in data_from.objects if name in import_objects]
    #     print('These are the objs: ', data_to.objects)
    #
    #     # Objects have to be linked to show up in a scene
    #     for obj in bpy.data.objects:
    #         if obj.name in import_objects:
    #             link(obj,collection="Mountains")


def get_image(src):
    return bpy.data.images.load(os.path.join(IMG_DIR, src))
