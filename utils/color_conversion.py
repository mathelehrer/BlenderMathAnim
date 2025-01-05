# !BPY
"""
Name: 'Color &lt;-&gt; HSV Cube'
Blender: 237
Group: 'Mesh'
Tooltip: '1 Selected: New mesh XYZ &gt; HSV,  2 Selected: write colour cube to original mesh.'
"""
from utils.constants import COLOR_PREFIXES, COLOR_NAMES, COLORS_SCALED

'''
RGB to HSV & HSV to RGB

The Hue/Saturation/Value model was created by A. R. Smith in 1978. 
It is based on such intuitive color characteristics as tint, 
shade and tone (or family, purety and intensity). 
The coordinate system is cylindrical, and the colors are defined inside a hexcone.
The hue value H runs from 0 to 360º. 
The saturation S is the degree of strength or purity and is from 0 to 1. 
Purity is how much white is added to the color, 
so S=1 makes the purest color (no white). 
Brightness V also ranges from 0 to 1, where 0 is the black.

There is no transformation matrix for RGB/HSV conversion, but the algorithm follows:
'''


#### deal with colors ####

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



# r,g,b values are from 0 to 1
# h = [0,360], s = [0,1], v = [0,1]
#        if s == 0, then h = -1 (undefined)

def rgb2hsv(R, G, B):
    # min, max, delta;
    min_rgb = min(R, G, B)
    max_rgb = max(R, G, B)
    V = max_rgb

    delta = max_rgb - min_rgb
    if not delta:
        H = 0
        S = 0
        V = R  # RGB are all the same.
        return H, S, V

    elif max_rgb:  # != 0
        S = delta / max_rgb
    else:
        R = G = B = 0  # s = 0, v is undefined
        S = 0
        H = 0  # -1
        return H, S, V

    if R == max_rgb:
        H = (G - B) / delta  # between yellow & magenta
    elif G == max_rgb:
        H = 2 + (B - R) / delta  # between cyan & yellow
    else:
        H = 4 + (R - G) / delta  # between magenta & cyan

    H *= 60  # degrees
    if H < 0:
        H += 360

    return H, S, V


#
def hsv2rgb(H, S, V):
    """
    >>> hsv2rgb(360,1,1) # cyan
    (1, 0.0, 0)

    >>> hsv2rgb(0,1,1)
    (1, 0.0, 0)

    >>> hsv2rgb(180,1,1) # cyan
    (0, 1.0, 1)

    >>> hsv2rgb(90,1,1)
    (0.5, 1, 0)

    >>> hsv2rgb(270,1,1)
    (0.5, 0, 1)

    """

    if not S:  # S == 0
        # achromatic (grey)
        # R = G = B = V
        return V, V, V  # RGB == VVV

    H /= 60;  # sector 0 to 5
    i = int(H)  # round down to int. in C its floor()
    f = H - i  # factorial part of H
    p = V * (1 - S)
    q = V * (1 - S * f)
    t = V * (1 - S * (1 - f))

    if i%6 == 0:
        R, G, B = V, t, p
    elif i%6 == 1:
        R, G, B = q, V, p
    elif i%6 == 2:
        R, G, B = p, V, t
    elif i%6 == 3:
        R, G, B = p, q, V
    elif i%6 == 4:
        R, G, B = t, p, V
    else: # 5
        R, G, B = V, p, q
    return R, G, B

