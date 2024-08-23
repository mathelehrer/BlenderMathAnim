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
    'gray_1',
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


# Warning: each color scheme has to provide colors for these COLOR_NAMES, otherwise it will yield errors
if color_scheme == -1:
    # https://digitalsynopsis.com/design/color-schemes-palettes-combinations/  four seasons
    COLORS = [
        [10, 10, 10, 1],  # black background
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
    ]

elif color_scheme == 0:
    # https://digitalsynopsis.com/design/color-schemes-palettes-combinations/  four seasons
    COLORS = [
        [0, 0, 0, 1],  # black background
        [247, 220, 104, 1],  # milky yellow
        [46, 149, 153, 1],  # turquoise
        [244, 108, 63, 1],  # orange
        [167, 34, 111, 1],  # pink
        [255, 255, 255, 1],  # white
        [225, 0, 32, 1],  # custom 1 # the values are adjusted to incorporate the gamma factor of 1.2 in the end
        [249, 189, 179, 1],  # custom 2
        [234, 75, 77, 1],  # custom 3
        [243, 153, 143, 1],  # custom 4
        [239, 115, 107, 1],  # custom 5
    ]

elif color_scheme == 4:  # UCSF
    # https://identity.ucsf.edu/print-digital/digital-colors
    COLORS = [
        [5, 32, 73, 1],  # Dark blue
        [255, 255, 255, 1],  # White
        [255, 221, 0, 1],  # Yellow
        [80, 99, 128, 1],  # Light Navy
        [47, 51, 54, 1],  # Two darker than first video
        [0, 0, 0, 1],  # Black
        [113, 111, 178, 1],  # Light purple
        [180, 185, 191, 1],  # Dark gray
        [209, 211, 211, 1],  # Light gray
    ]
    COLOR_NAMES = None

elif color_scheme == 5:
    # https://adrian.simionov.io/aws/2020/04/24/aws-color-palette.html
    COLORS = [
        [35, 47, 60, 1],  # dark blue background
        [143, 168, 194, 1],  # light blue foreground
        [92, 191, 168, 1],  # turquoise
        [253, 152, 49, 1],  # orange
        [252, 85, 139, 1],  # pink
        [249, 90, 83, 1],  # red
        [157, 113, 249, 1],  # purple
        [78, 134, 249, 1],  # blue
        [34, 163, 196, 1],  # cyan
        [112, 171, 74, 1],  # green
        [250, 250, 250, 1],  # white
        [59, 76, 89, 1],  # background_alt
        [78, 88, 98, 1],  # background_alt2
        [73, 86, 90, 1],  # background_alt3
    ]

elif color_scheme == 1:
    # Coolors Exported Palette - https://coolors.co/393e41-f3f2f0-3e7ea0-ff9400-e7e247
    COLORS = [
        [57, 62, 65, 1],
        # [211, 208, 203, 1],
        [243, 242, 240, 1],
        [62, 126, 160, 1],
        [255, 148, 0, 1],
        # [232, 225, 34, 1],
        [231, 226, 71, 1],
        # [106, 141, 115, 1]
        [215, 38, 61, 1]
        # [255, 0, 0, 1]
    ]

elif color_scheme == 2:  # Main. Why isn't #1 main? Because your face.
    COLORS = [
        # [42, 46, 48, 1], #Three darker than first video
        [47, 51, 54, 1],  # Two darker than first video
        # [211, 208, 203, 1],
        [243, 242, 240, 1],
        [62, 126, 160, 1],
        [255, 148, 0, 1],
        # [232, 225, 34, 1],
        [231, 226, 71, 1],
        # [106, 141, 115, 1]
        [214, 59, 80, 1],
        # [255, 0, 0, 1]
        [105, 143, 63, 1],
        [219, 90, 186, 1],
        [145, 146.5, 147, 1],  # Gray from averaging 1 and 2
        [0, 0, 0, 1]
    ]


elif color_scheme == 3:
    # Coolors Exported Palette - coolors.co/191308-bbd8b3-f3b61f-48a0c9-72120d
    COLORS = [
        [1, 33, 31, 1],
        [255, 237, 225, 1],
        [32, 164, 243, 1],
        [255, 51, 102, 1],
        [234, 196, 53, 1],
        [215, 38, 61, 1]
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
RES_SVG_DIR = os.path.join(RES_DIR, "svg")
RES_HDRI_DIR = os.path.join(RES_DIR, 'hdri')
MEDIA_DIR = os.path.join(LOC_FILE_DIR, "media")
TEX_TPL_DIR = os.path.join(RES_DIR, "tex")
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
OSL_DIR = os.path.join(RES_DIR, "osl")
PRIMITIVES_DIR = os.path.join(RES_DIR, 'blend/primitives')

print(os.getcwd())

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
