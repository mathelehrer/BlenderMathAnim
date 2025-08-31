import os
from copy import deepcopy
import numpy as np
from mathutils import Vector

'''
Quality
'''
RENDER_QUALITY = 'final'
# 'medium' or higher turns on expression morphing
# which takes a few seconds per run
if RENDER_QUALITY == 'final':
    RESOLUTION_PERCENTAGE = 100
    LIGHT_SAMPLING_THRESHOLD = 0.01
    SAMPLE_COUNT = 64
    RENDER_QUALITY = 'high'
    RENDER_TILE_SIZE = 256
    # The 'final' stuff is over, so just setting to 'high' for rest of code
else:
    RESOLUTION_PERCENTAGE = 30
    LIGHT_SAMPLING_THRESHOLD = 0.0  # For my scenes, it seems there's so little
    # ray bouncing that the cutoff test actually
    # takes longer than letting the light keep going
    SAMPLE_COUNT = 64
    RENDER_TILE_SIZE = 512

if RENDER_QUALITY == 'high':
    LIGHTS_TO_A_SIDE = 1
    LIGHT_TYPE = 'SUN'
    ICO_SUBDIVISIONS = 6
    CONTROL_POINTS_PER_SPLINE = 40  # TODO: figure out the threshold for noticing a difference
    PARTICLES_PER_MESH = 300000  # Could be smaller if morphing smaller objects
    # Could even be a bobject of scale
    # Or number of other objects

else:
    LAMPS_TO_A_SIDE = 1
    LAMP_TYPE = 'SUN'
    ICO_SUBDIVISIONS = 2
    CONTROL_POINTS_PER_SPLINE = 30
    PARTICLES_PER_MESH = 1000

    RESOLUTION_PERCENTAGE = 30
    LIGHT_SAMPLING_THRESHOLD = 0.1
    SAMPLE_COUNT = 64

'''
My Quality settings
'''

# QUALITY = 'exceptional'
QUALITY = 'final'

if QUALITY == 'low':
    RESOLUTION_PERCENTAGE = 33
    CONTROL_POINTS_PER_SPLINE = 5
    FRAME_RATE = 15
elif QUALITY == 'high':
    RESOLUTION_PERCENTAGE = 50
    CONTROL_POINTS_PER_SPLINE = 10
    FRAME_RATE = 30
elif QUALITY == 'final':
    RESOLUTION_PERCENTAGE = 100
    CONTROL_POINTS_PER_SPLINE = 20
    FRAME_RATE = 60
    LIGHT_SAMPLING_THRESHOLD = 0.01
elif QUALITY == 'exceptional':
    RESOLUTION_PERCENTAGE = 100
    CONTROL_POINTS_PER_SPLINE = 20
    FRAME_RATE = 120
    LIGHT_SAMPLING_THRESHOLD = 0.005

'''
global fields
'''
IMPORTED_OBJECTS = []  # array that keeps track of all imported objects, to avoid that an imported object is referenced twice
IMPORTED_CURVES = []
'''
Colors
'''
color_scheme = -1
COLOR_NAMES = [
    'background',
    'text',
    'drawing',
    'example',
    'important',
    'joker',
    'custom1',
    'custom2',
    'custom3',
    'custom4',
    'custom5',
    'code_builtin',
    'code_builtin2',
    'code_override',
    'code_self',
    'code_number',
    'code_string',
    'code_function',
    'code_keyword',
    'some_logo_green',
    'some_logo_blue',
    'x12_color',
    'x13_color',
    'x14_color',
    'x15_color',
    'x23_color',
    'x24_color',
    'x25_color',
    'x34_color',
    'x35_color',
    'x45_color',
    'green',
    'red',
    'blue',
    'orange',
    'yellow',
    'magenta',
    'cyan',
    'mega1',
    'mega2',
    'mega3',
    'mega4',
    'mega5',
    'mega6',
    'mega7',
    'mega8',
    'mega9',
    'mega10',
    'mega11',
    'mega12',
    'billiard_1',
    'billiard_2',
    'billiard_3',
    'billiard_4',
    'billiard_5',
    'billiard_6',
    'billiard_7',
    'billiard_8',
    'billiard_0',
    'terminal_green',
    'gray_1', # gray always has to stay last in this list
    'gray_2',
    'gray_3',
    'gray_4',
    'gray_5',
    'gray_6',
    'gray_7',
    'gray_8',
    'gray_9',
]

COLOR_PREFIXES =['plastic','glass','fake_glass']
# special materials to make sure that they are not overridden by default brightness, roughness, etc settings
# if you want to change single properties of these special materials use the override_material=True flag
SPECIALS = ['screen','scatter_volume','gold','silver','screen','wood','sand','metal','glass','mirror','silk','fake_glass','plastic']

DATA_TYPES = ['FLOAT', 'INT', 'FLOAT_VECTOR', 'FLOAT_COLOR', 'BYTE_COLOR', 'BOOLEAN', 'FLOAT2', 'INT8', 'QUATERNION', 'FLOAT4X4']

# Warning: each color scheme has to provide colors for these COLOR_NAMES, otherwise it will yield errors
if color_scheme == -1:
    # https://digitalsynopsis.com/design/color-schemes-palettes-combinations/  four seasons
    COLORS = [
        [0, 0, 0, 1],  # black background
        [225, 225, 225, 1],  # white for text
        [86, 180, 233, 1],  # sky blue for drawing
        [240, 228, 66, 1],  # yellow for example
        [213, 94, 10, 1],  # vermillion for important
        [10, 158, 115, 1],  # joker
        [230, 10, 52, 1],  # custom 1 # the values are adjusted to incorporate the gamma factor of 1.2 in the end
        [250, 200, 190, 1],  # custom 2
        [236, 93, 96, 1],  # custom 3
        [245, 167, 158, 1],  # custom 4

        [241, 132, 125, 1],  # custom 5
        [int(80*255/100),int(47.2*255/100),int(12.4*255/100),1],  # code_builtin
        [int(49.8*255/100),int(50.9*255/100),int(75.4*255/100),1],  # code_builtin
        [int(70*255/100),int(0*255/100),int(72*255/100),1],  # code_override
        [int(58.8*255/100),int(32.2*255/100),int(56.6*255/100),1],  # code_self
        [int(40.6*255/100),int(59.8*255/100),int(74.7*255/100),1],  # code_number
        [int(41.5*255/100),int(53.5*255/100),int(33.2*255/100),1],  # code_string
        [int(95.2*255/100),int(74.6*255/100),int(37.5*255/100),1],  # code_function
        [int(67.5*255/100),int(27.7*255/100),int(9.5*255/100),1],  # code_keyword
        [int(141),int(171),int(127),1],  # some_logo_green

        [int(88),int(123),int(127),1],  # some_logo_blue
        [158,1,66,1],#x12
        [50, 136, 189, 1],#x13
        [94, 79, 162, 1],  # x14
        [244,109,67,1],#x15
        [171,221,164,1],#x23
        [213,62,79,1],#x24
        [230,245,152,1],#x25
        [254,224,139,1],# X34
        [253,174,97,1],#x35

        [102,194,165,1],#x45
        [0,255,0,1],#green
        [70,0,0,1],# red
        [0,0,255,1],# blue
        [255,30,8,1],# orange
        [200,200,0,1],# yellow
        [255,0,255,1],# magenta
        [0,255,255,1], # cyan
        [187,220,126,1],# mega1
        [251,107,-12,1],# mega2

        [194,194,194,1],# mega3
        [241,157,192,1],# mega4
        [255,251,158,1],# mega5
        [118,199,241,1],# mega6
        [-5,102,-11,1],# mega7
        [255,255,255,1],# mega8
        [95,80,156,1],# mega9
        [250,196,-15,1],# mega10
        [2,56,187,1],# mega11
        [220,31,22,1],# mega12

        [242, 211, 0, 1],# billiard 1
        [0, 92, 171, 1],# billiard 2
        [205, 32, 44, 1],# billiard 3
        [112, 48, 160, 1], # billiard 4
        [241, 101, 34, 1], # billiard 5
        [0, 132, 70, 1], # billiard 6
        [125, 60, 43, 1],# billiard 7
        [0, 0, 0, 1], # billiard 8
        [255,255,255,0],# billiard 0

        [78, 154, 6,1], # terminal_green
    ]

# add gray values
for i in range(1, 10):
    COLORS.append([i * 25, i * 25, i * 25, 1])

COLORS_SCALED = []
for color in COLORS:
    color_scaled = deepcopy(color)
    for i in range(3):
        color_scaled[i] /= 255
        # color_scaled[i] = color_scaled[i] ** 2.2
    COLORS_SCALED.append(color_scaled)

if COLOR_NAMES is not None:
    zip_iterator = zip(COLOR_NAMES, COLORS_SCALED)
    colors = dict(zip_iterator)

'''
File and directory constants
'''
LOC_FILE_DIR = os.getcwd()
###
THIS_DIR = os.path.join(LOC_FILE_DIR,"..")
RES_DIR = os.path.join(THIS_DIR, "files")
RES_DIR2 = os.path.join(THIS_DIR, "files2")
RES_XML = os.path.join(RES_DIR2, "xml")
SHADER_XML=os.path.join(RES_XML,"textures")
RES_SVG_DIR = os.path.join(RES_DIR, "svg")
RES_HDRI_DIR = os.path.join(RES_DIR, 'hdri')
OBJ_DIR = os.path.join(RES_DIR, "obj")
MEDIA_DIR = os.path.join(LOC_FILE_DIR, "media")
TEX_TPL_DIR = os.path.join(RES_DIR, "tex")
FONT_DIR = os.path.join(RES_DIR, "fonts")
BLEND_TPL_DIR = os.path.join(RES_DIR, "blend")
BLEND_PRESET_DIR = os.path.join(BLEND_TPL_DIR, "presets")
APPEND_DIR = os.path.join(LOC_FILE_DIR,"appends")
BLEND_FRM_RATE_DIR = os.path.join(BLEND_PRESET_DIR, "framerate")
TEX_DIR = os.path.join(MEDIA_DIR, "tex")
SVG_DIR = os.path.join(MEDIA_DIR, "svg")
IMG_DIR = os.path.join(MEDIA_DIR, "raster")
VID_DIR = os.path.join(MEDIA_DIR,"vids")
BLEND_DIR = os.path.join(MEDIA_DIR, "blend")
FINAL_DIR = os.path.join(BLEND_DIR, "final")
RENDER_DIR = "/filme/working_dir/"
DATA_DIR = os.path.join(LOC_FILE_DIR, 'data')
GLOBAL_DATA_DIR=os.path.join(RES_DIR,"data")
GLOBAL_DATA_DIR2=os.path.join(RES_DIR2,"data")
OSL_DIR = os.path.join(RES_DIR, "osl")
PRIMITIVES_DIR = os.path.join(RES_DIR, 'blend/primitives')

for folder in [MEDIA_DIR, TEX_DIR, SVG_DIR, BLEND_DIR, RES_DIR, TEX_TPL_DIR, BLEND_TPL_DIR, IMG_DIR,
               RENDER_DIR, DATA_DIR,APPEND_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("created folder: " + folder)

TEX_TEXT_TO_REPLACE = "YourTextHere"
TEMPLATE_TEX_FILE = os.path.join(TEX_TPL_DIR, "template_arial.tex")
TEMPLATE_TEXT_FILE = os.path.join(TEX_TPL_DIR,"template_arial_text.tex")

'''
Camera and lighting constants
'''
CAMERA_LOCATION = (0, -20, 0)
CAMERA_ANGLE = (np.pi / 2, 0, 0)

'''
Text constants
'''
SPACE_BETWEEN_EXPRESSIONS = 0.45  # 0.1 For text svgs  #0.45 For actual tex_bobjects
TEX_LOCAL_SCALE_UP = 260  # Value that makes line height about 1 Blender Unit

'''
Animation constants
'''

OBJECT_APPEARANCE_TIME = 1
PARTICLE_APPEARANCE_TIME = 1
DEFAULT_MORPH_TIME = OBJECT_APPEARANCE_TIME
DEFAULT_ANIMATION_TIME = OBJECT_APPEARANCE_TIME
MORPH_PARTICLE_SIZE = 0.03
FLOOR_APPEARANCE_TIME = OBJECT_APPEARANCE_TIME  # time for floor to appear in a new world
TEXT_APPEARANCE_TIME = OBJECT_APPEARANCE_TIME
DEFAULT_SCENE_DURATION = 10
DEFAULT_SCENE_BUFFER = 0  # This was used when multiple scenes were in blender at
# once, which was basically never, and will never be.
# Could just delete.

'''
Graph constants
'''
AXIS_WIDTH = 0.05
AXIS_DEPTH = 0.01
ARROW_SCALE = [0.3, 0.3, 0.4]
GRAPH_PADDING = 1
CURVE_WIDTH = 0.04
if RENDER_QUALITY == 'high':
    PLOTTED_POINT_DENSITY = 30
else:
    PLOTTED_POINT_DENSITY = 20
CURVE_Z_OFFSET = 0.01
AUTO_TICK_SPACING_TARGET = 2  # Blender units
HIGHLIGHT_POINT_UPDATE_TIME = OBJECT_APPEARANCE_TIME

'''
Manim like constants
'''

UP = Vector([0, 0, 1])
DOWN = -UP
RIGHT = Vector([1, 0, 0])
LEFT = -RIGHT
BACK = Vector([0, 1, 0])
FRONT = -BACK

SMALL_BUFF = 0.25
