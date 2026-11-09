import inspect
import os
import time
from copy import deepcopy

import bpy

from appearance.textures import make_basic_material, make_creature_material, make_translucent_material, \
    make_fake_glass_material, make_plastic_material, make_checker_material, make_mirror_material, make_sand_material, \
    make_gold_material, make_silver_material, make_screen_material, make_marble_material, make_metal_materials, \
    make_wood_material, make_scattering_material, make_silk_material, make_magnet_material, make_sign_material, \
    make_cloud_material, make_six_color_ramp_material, make_eevee_glass_material
from interface import ibpy
from interface.interface_constants import blender_version
from perform.render import render_with_skips
from utils.constants import DEFAULT_SCENE_BUFFER, LIGHT_TYPE, CAMERA_LOCATION, CAMERA_ANGLE, FRAME_RATE, COLORS_SCALED, \
    DEFAULT_SCENE_DURATION, SAMPLE_COUNT, LIGHT_SAMPLING_THRESHOLD, RESOLUTION_PERCENTAGE, RENDER_DIR, FILM_DIR, \
    BLEND_FRM_RATE_DIR, COLOR_NAMES, COLORS, FONT_DIR, ensure_runtime_directories
from utils.kwargs import get_from_kwargs

OLDSCHOOL = "AcPlus IBM VGA 8x16 Regular"

class Scene(object):
    def __init__(self, **kwargs):
        ensure_runtime_directories()
        # name of the video, used for the render directory /filme/<video_name>_<scene_name>/
        # default: the directory of the script that defines the subclass, without the "video_" prefix
        self.video_name = kwargs.pop('video_name', None) or self.default_video_name()
        self.scene_name = ""
        self.kwargs = kwargs;
        self.is_created = False
        try:
            total_duration = 0
            for sub, attrs in self.sub_scenes.items():
                print(sub, " ", attrs)
                total_duration += attrs['duration']

            self.duration = total_duration
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            raise Warning('Must define self.sub_scenes in subclass of Scene')

        self.set_sub_scene_timing()

    def default_video_name(self):
        directory = os.path.basename(os.path.dirname(os.path.abspath(inspect.getfile(self.__class__))))
        if directory.startswith("video_"):
            directory = directory[len("video_"):]
        return directory

    def render_directory(self, name=None):
        """
        Directory for the rendered images: /filme/<video_name>_<scene_name>/
        It is created, when it does not exist.

        :param name: name of the sub scene, defaults to the last played sub scene
        :return: the path of the directory
        """
        if name is None:
            name = self.scene_name
        folder = self.video_name + "_" + name if name else self.video_name
        path = os.path.join(FILM_DIR, folder)
        os.makedirs(path, exist_ok=True)
        return path

    def set_sub_scene_timing(self):
        start = 0
        for sub, attrs in self.sub_scenes.items():
            attrs['start'] = start
            attrs['end'] = start + attrs['duration']
            start = attrs['end']

    def play(self,name=None,resolution=[1920,1080],start_at_zero=False):
        if not name:
            start=0
            duration=self.duration
        else:
            if name in self.sub_scenes and not start_at_zero:
                sub = self.sub_scenes[name]
                start=sub['start']
                duration=sub['end']-start
            else:
                start = 0
                subs = self.sub_scenes[name]
                duration = subs['duration']

        self.scene_name = name if name else ""
        initialize_blender(start,duration,resolution=resolution,render_dir=self.render_directory(),
                           clear_previous=self.is_created, **self.kwargs)
        #initialize_blender(total_duration=self.duration, **self.kwargs)
        if name:
            if hasattr(self, name):
                getattr(self, name)()
        print("Scene finished in time range ",start," to ",start+duration)
        if hasattr(self,"t0"):
            print("The animation timer stopped at ",self.t0)

    def create(self,name="",resolution=[1920,1080],start_at_zero=False):
        start = time.time()
        self.play(name,resolution=resolution,start_at_zero=start_at_zero)
        self.is_created = True
        self.save(name)
        end = time.time()
        print(end - start," seconds elapsed.")

    def parse_selection(self, selection):
        """
        Translate the user's choice into a list of sub scene names.

        Accepted forms (indices refer to the numbered list of sub scenes, ranges are inclusive):
          ""            -> sub scene 0
          "3"           -> sub scene 3
          "0-9", "[0-9]"-> sub scenes 0 to 9
          "1,4,7-9"     -> sub scenes 1, 4, 7, 8, 9
          "pixel_lines" -> the sub scene with this name

        :param selection: the string typed by the user
        :return: list of sub scene names in the order given
        """
        names = list(self.sub_scenes.keys())
        selection = selection.strip().strip("[]").strip()
        if len(selection) == 0:
            return names[:1]
        selected = []
        for part in selection.split(","):
            part = part.strip()
            if part in self.sub_scenes:
                selected.append(part)
            elif "-" in part:
                first, last = (int(bound) for bound in part.split("-", 1))
                step = 1 if last >= first else -1
                selected += [names[i] for i in range(first, last + step, step)]
            else:
                selected.append(names[int(part)])
        return selected

    def choose_and_create(self, resolution=[1920, 1080], start_at_zero=True):
        """
        List the sub scenes, ask for a selection and create one blender file per selected sub scene.
        See parse_selection for the accepted input.

        Example (the __main__ block of a video script):
            example = EllipticCurvesScene()
            example.choose_and_create(resolution=[1920, 1080], start_at_zero=True)
        """
        names = list(self.sub_scenes.keys())
        for i, name in enumerate(names):
            print(i, name)
        if len(names) == 1:
            selected = names
        else:
            print("Usage: <enter> -> 0 | 3 -> sub scene 3 | 0-9 or [0-9] -> sub scenes 0 to 9 (inclusive)"
                  " | 1,4,7-9 -> a list | pixel_lines -> by name")
            print("A selection of several sub scenes creates one blender file per sub scene.")
            selected = self.parse_selection(input("Choose scene(s):"))
        print("Your choice: ", selected)
        for name in selected:
            self.create(name=name, resolution=resolution, start_at_zero=start_at_zero)

    def render(self,debug=False,overwrite=False):
        if not self.is_created:
            self.create()
        start = ibpy.start_frame()
        end = ibpy.end_frame()
        render_with_skips(start, end,debug,overwrite,filepath=self.render_directory())

    def final_render(self,name="",debug=True,overwrite=False):
        self.load(name)
        start = ibpy.start_frame()
        end = ibpy.end_frame()
        render_with_skips(start, end, debug, overwrite, filepath=self.render_directory(name))

    def save(self,name):
        if not self.is_created:
            self.create()
        ibpy.save(self.__class__.__name__+"_"+name)

    def load(self,name):
        ibpy.load_file(self.__class__.__name__+"_"+name)


def is_scene(obj):
    # print('checking scene')
    # if "TextScene" in str(obj):
    if not inspect.isclass(obj):
        # print('  not class')
        return False
    if not issubclass(obj, Scene):
        print(obj)
        print('  not subclass of scene')
        return False
    if obj == Scene:
        print(obj)
        print('  is scene class itself')
        return False
    return True


#def initialize_blender(total_duration=DEFAULT_SCENE_DURATION, clear_blender=True, vertical=False, **kwargs):
def initialize_blender(start,duration, short=False,resolution=[1920,1080],clear_blender=True, vertical=False, render_dir=RENDER_DIR, clear_previous=False, **kwargs):
    if clear_blender:  # clear objects and materials
        print('Clearing Blender data')
        for bpy_data_iter in (
                bpy.data.objects,
                bpy.data.meshes,
                bpy.data.cameras,
                bpy.data.curves,
                bpy.data.materials,
                bpy.data.particles,
                bpy.data.worlds):
            for id_data in bpy_data_iter:
                bpy_data_iter.remove(id_data)
    if clear_previous:  # collections and node groups of a sub scene created before in the same session
        # the default collection "Collection" of the startup file stays, objects are linked to it and
        # ibpy.un_link(..., collection="Collection") relies on it
        for collection in list(bpy.context.scene.collection.children_recursive):
            if collection.name != "Collection":
                bpy.data.collections.remove(collection)
        if "Collection" not in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.link(
                bpy.data.collections.get("Collection") or bpy.data.collections.new("Collection"))
        for node_group in list(bpy.data.node_groups):
            bpy.data.node_groups.remove(node_group)


    scn = bpy.context.scene
    scn.render.engine = 'CYCLES'
    scn.cycles.device = 'GPU'
    scn.cycles.denoising_use_gpu=True
    scn.cycles.samples = SAMPLE_COUNT
    scn.cycles.preview_samples = SAMPLE_COUNT
    scn.cycles.light_sampling_threshold = LIGHT_SAMPLING_THRESHOLD
    scn.cycles.transparent_max_bounces = 40
    scn.render.resolution_percentage = RESOLUTION_PERCENTAGE
    scn.render.use_compositing = False
    scn.render.use_sequencer = False
    scn.render.image_settings.file_format = 'PNG'

    scn.render.resolution_x = resolution[0]
    scn.render.resolution_y = resolution[1]

    # set view to Material view
    area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
    space = next(space for space in area.spaces if space.type == 'VIEW_3D')
    space.shading.type = 'MATERIAL'  # set the viewport shading
    # set view to rendered view
    space.shading.type = 'RENDERED'  # set the viewport shading
    space.shading.use_compositor='ALWAYS'

    bpy.data.scenes["Scene"].render.filepath = os.path.join(render_dir, "")

    if vertical:
        scn.render.resolution_x = resolution[1]
        scn.render.resolution_y = resolution[0]

    # Apparentlly 16-bit color depth pngs don't convert well to mp4 in Blender.
    # It gets all dark. 8-bit it is.
    # BUT WAIT. I can put stacks of pngs straight into premiere.
    scn.render.image_settings.color_depth = '16'
    scn.render.image_settings.color_mode = 'RGBA'
    if blender_version()<(5,0):
        scn.cycles.film_transparent = False
    else:
        scn.render.film_transparent = False

    # Set to FRAME_RATE
    bpy.ops.script.execute_preset(
        filepath=os.path.join(BLEND_FRM_RATE_DIR,str(FRAME_RATE)+".py"),
        menu_idname="RENDER_MT_framerate_presets"
    )

    # The line below makes it so Blender doesn't apply gamma correction. For some
    # reason, Blender handles colors differently from how every other program
    # does, and it's terrible. Why.
    # But this fixes it. Also, the RGB values you see in Blender
    # will be wrong, because the gamma correction is still applied when the color
    # is defined, but setting view_transform to 'Raw' undoes the correction in
    # render.

    # scn.view_settings.view_transform = 'Raw'
    # scn.view_settings.gamma = 1.2

    # since rubik's cube there is a different choice
    scn.view_settings.view_transform = 'Filmic'
    scn.view_settings.look = 'High Contrast'

    scn.gravity = (0, 0, -9.81)


    # setup simple background with light path for reflections
    bpy.ops.world.new()
    world = bpy.data.worlds[-1]
    scn.world = world
    nodes = world.node_tree.nodes
    mixer = nodes.new(type='ShaderNodeMix')
    mixer.data_type='RGBA'
    light_path = nodes.new(type='ShaderNodeLightPath')
    world_out = nodes.get("World Output")
    color = nodes.new(type='ShaderNodeRGB')
    world.node_tree.links.new(light_path.outputs['Is Camera Ray'], mixer.inputs['Factor'])
    world.node_tree.links.new(color.outputs['Color'], mixer.inputs['A'])
    world.node_tree.links.new(mixer.outputs['Result'], world_out.inputs['Surface'])
    color.outputs['Color'].default_value = COLORS_SCALED[0] # background color by default

    define_materials()

    # set up timeline
    bpy.data.scenes["Scene"].frame_start = start*FRAME_RATE
    bpy.data.scenes["Scene"].frame_end = (start+duration) * FRAME_RATE - 1
    bpy.context.scene.frame_set(0)

    # create camera and light
    ibpy.add_camera(location=CAMERA_LOCATION,rotation=CAMERA_ANGLE)
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))

    light_parent = bpy.context.object
    light_parent.name = 'lights'

    light_height = get_from_kwargs(kwargs, 'light_height', 35)
    light_location = get_from_kwargs(kwargs, 'light_location', (0, -20, light_height))
    light_energy = get_from_kwargs(kwargs, 'light_energy', 1)
    light_type = get_from_kwargs(kwargs, 'light_type', LIGHT_TYPE)
    bpy.ops.object.light_add(type=light_type, location=light_location)
    light = bpy.context.active_object
    light.data.energy = light_energy
    ibpy.set_shadow(True)
    light_shine_to = get_from_kwargs(kwargs, 'light_shine_to', [0, 0, 0])
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=light_shine_to)
    light_empty = bpy.context.active_object
    light_empty.parent = light_parent
    constraint = light.constraints.new(type='TRACK_TO')
    constraint.target = light_empty
    light.parent = light_parent
    light.matrix_parent_inverse = light.parent.matrix_world.inverted()

    # light.data.node_tree.nodes[1].inputs[1].default_value = 1.57
    # light.data.shadow_soft_size = 3

    # Sets view to look through camera.
    # maybe not needed

    area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
    area.spaces[0].region_3d.view_perspective = 'CAMERA'

    # transparent background
    if 'transparent' in kwargs:
        transparent = kwargs.pop('transparent')
    else:
        transparent = False
    scn.render.film_transparent=transparent

    # load fonts

    bpy.ops.font.open(filepath=os.path.join(FONT_DIR, "Symbola/Symbola.ttf"))
    bpy.ops.font.open(filepath=os.path.join(FONT_DIR, "arial/ARIBLK.TTF"))
    bpy.ops.font.open(filepath=os.path.join(FONT_DIR, "Engineering_Plot/Engplot.TTF"))
    bpy.ops.font.open(filepath=os.path.join(FONT_DIR, "OldSchool/OldSchool.ttf"))


def get_total_duration(scenes):
    # scenes is a list of (name, object) pairs
    duration = 0
    for scene in scenes:
        duration += scene[1].duration + DEFAULT_SCENE_BUFFER
    return duration


def define_materials():
    if 'default' not in bpy.data.materials:
        mat = bpy.data.materials.new(name='default')
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.8, 1, 1, 1)

    for i, col_name in enumerate(COLOR_NAMES):
        name = col_name
        col = COLORS[i]
        make_basic_material(rgb=deepcopy(col), name=name)
        name = 'creature_color' + str(i + 1)
        make_creature_material(rgb=deepcopy(col), name=name)
        name = 'fake_glass_' + col_name
        make_fake_glass_material(rgb=deepcopy(col), name=name)
        name = 'plastic_' + col_name
        make_plastic_material(rgb=deepcopy(col), name=name)
        name = 'eevee_glass_' + col_name
        make_eevee_glass_material(rgb=deepcopy(col), name=name)

    # create checker material
    # it's outdated to preload them
    # it should be converted into functions that create materials on demand and should be called with **kwargs and get_texture()
    make_mirror_material()
    make_sand_material()
    make_gold_material()
    make_cloud_material()
    make_silver_material()
    make_screen_material()
    make_marble_material()
    make_metal_materials()
    make_wood_material()
    make_scattering_material()
    make_silk_material()
    make_magnet_material()
    make_sign_material()
    make_six_color_ramp_material()