import colorsys
import os
from copy import deepcopy
from random import random

import bpy
import numpy as np
from sympy import Symbol, re, im

from geometry_nodes.nodes import make_function
from interface import ibpy
from interface.ibpy import customize_material, make_alpha_frame, create_group_from_vector_function, \
    Vector, set_material, get_color_from_string, create_iterator_group, get_obj, get_color, animate_sky_background
from interface.interface_constants import TRANSMISSION, SPECULAR, EMISSION
from mathematics.parsing.parser import ExpressionConverter
from mathematics.spherical_harmonics import SphericalHarmonics
from physics.constants import temp2rgb, type2temp
from shader_nodes.shader_nodes import TextureCoordinate, Mapping, ColorRamp, AttributeNode, HueSaturationValueNode, \
    MathNode, MixRGB, InputValue, GradientTexture, ImageTexture, SeparateXYZ
from utils.constants import COLORS, COLORS_SCALED, COLOR_NAMES, IMG_DIR
from utils.kwargs import get_from_kwargs


def convert_strings_to_colors(color_names):
    if isinstance(color_names, str):
        return get_color_from_name(color_names)
    else:
        return_list = []
        for l in color_names:
            return_list.append(convert_strings_to_colors(l))
        return return_list


def get_color_from_name(color_name):
    return COLORS_SCALED[COLOR_NAMES.index(color_name)]


def apply_material(obj, col, shading=None, recursive=False, type_req=None, intensity=None, **kwargs):
    """

    intensity and so on
    :param shading:
    :param obj:
    :param col:
    :param recursive:
    :param type_req:
    :param intensity:
    :return:
    """

    if obj.type not in ['EMPTY', 'ARMATURE']:
        if type_req is None or obj.type == type_req:
            if col == 'vertex_color':
                material = vertex_color_material()
            elif col is None:
                if 'colors' in kwargs:
                    colors = kwargs.pop('colors')
                    for col, slot in zip(colors, obj.material_slots):
                        slot.material = ibpy.get_material(col, **kwargs)

                    material = obj.material_slots[0].material  # only the first material can be customized further
                else:
                    material = get_default_material().copy()
            elif isinstance(col, str):
                material = ibpy.get_material(col, **kwargs)
            elif callable(col):
                material = col(**kwargs)

            if shading is None:
                obj.active_material = material
            else:
                obj.active_material = shade_material(material, shading)

            if 'brighter' in kwargs:
                brighter = kwargs['brighter']
            else:
                brighter = 0
            if brighter:
                obj.active_material = light_up_material(material, brighter)

    if recursive:
        for child in obj.children:
            apply_material(child, material, recursive=recursive, type_req=type_req,**kwargs)

    if intensity is not None and 'trans' in material:
        nodes = obj.active_material.node_tree.nodes

        scat = nodes['Volume Scatter']
        absorb = nodes['Volume Absorption']
        emit = nodes['Emission']

        for node in [scat, absorb, emit]:
            node.inputs[1].default_value = intensity

    # material = obj.active_material
    for i, slot in enumerate(obj.material_slots):
        material = slot.material
        if material is not None:
            customize_material(material, **kwargs)

    # settings for eevee
    for slot in obj.material_slots:
        slot.material.blend_method = 'HASHED'
        slot.material.shadow_method = 'HASHED'

    if 'uv_alpha_frame' in kwargs:
        uv_alpha_frame = kwargs.pop('uv_alpha_frame')
        if 'alpha_thickness' in kwargs:
            alpha_thickness = kwargs.pop('alpha_thickness')
        else:
            alpha_thickness = 0.05
        if uv_alpha_frame:
            make_alpha_frame(material, alpha_thickness)

    if 'volume_absorption' in kwargs:
        ibpy.set_volume_absorption_of_material(material, value=kwargs.pop('volume_absorption'))

    if 'volume_scatter' in kwargs:
        ibpy.set_volume_scatter_of_material(material, value=kwargs.pop('volume_scatter'))

def vertex_color_material():
    vertex_color = bpy.data.materials.new(name="Vertex_Color")
    vertex_color.use_nodes = True
    nodes = vertex_color.node_tree.nodes
    links = vertex_color.node_tree.links
    bsdf = nodes["Principled BSDF"]
    col_att = nodes.new('ShaderNodeVertexColor')
    links.new(col_att.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(col_att.outputs['Alpha'], bsdf.inputs[TRANSMISSION])
    return vertex_color


def get_default_material():
    return bpy.data.materials['text']


def color2rgb(color):
    rgb = deepcopy(color)
    for i in range(3):
        rgb[i] /= 255
    return rgb


def phase2rgb2(phase):
    hue = phase / 2 / np.pi % 1
    col = colorsys.hsv_to_rgb(hue, 1, 1)
    return col[0], col[1], col[2], 1


def phase2rgb(phase, v=1, s=1):
    '''
    After a long phase of trial and error, this is a much nicer
    conversion from phase to rgb, since it make the red area much smaller
    and emphasizes all other colors

    :param phase:
    :param v:
    :param s:
    :return:
    '''
    h = (phase / 2 / np.pi % 1) * 360
    c = v * s
    x = c * (1 - np.abs(h / 60 % 2 - 1))
    m = v - c

    if 0 <= h < 60:
        rp = c
        gp = x
        bp = 0
    elif 60 <= h < 120:
        rp = x
        gp = c
        bp = 0
    elif 120 <= h < 180:
        rp = 0
        gp = c
        bp = x
    elif 180 <= h < 240:
        rp = 0
        gp = x
        bp = c
    elif 240 <= h < 300:
        rp = x
        gp = 0
        bp = c
    else:
        rp = c
        gp = 0
        bp = x

    map = linear_to_srgb(rp + m, gp + m, bp + m)
    return *map, 1


def linear_to_srgb(r, g, b):
    def srgb(c):
        a = .055
        if c <= .0031308:
            return c * 12.92
        else:
            return (1 + a) * c ** (1 / 2.4) - a

    return tuple(srgb(c) for c in (r, g, b))


def srgb_to_linear(r, g, b):
    def srgb(c):
        a = .055
        if c <= .04045:
            return c / 12.92
        else:
            return ((c + a) / (1 + a)) ** 2.4

    return tuple(srgb(c) for c in (r, g, b))


def clear_material(material):
    if material.node_tree:
        material.node_tree.links.clear()
        material.node_tree.nodes.clear()


def shade_material(material, shading):
    if isinstance(material, str):
        mat = ibpy.get_material().copy()
    else:
        mat = material

    bsdf = mat.node_tree.nodes["Principled BSDF"]

    color = bsdf.inputs['Base Color'].default_value
    up = 1.2
    down = 1 / up

    if shading == 'redder':
        color[0] *= up
        color[1] *= down
        color[2] *= down
    elif shading == 'greener':
        color[0] *= down
        color[1] *= up
        color[2] *= down
    elif shading == 'bluer':
        color[0] *= down
        color[1] *= down
        color[2] *= up
    elif shading == 'brighter':
        color[0] *= up
        color[1] *= up
        color[2] *= up
    elif shading == 'darker':
        color[0] *= down
        color[1] *= down
        color[2] *= down

    for i in range(3):
        color[i] = np.minimum(1, color[i])

    bsdf.inputs['Base Color'].default_value = color
    return mat


def light_up_material(material, brighter):
    if isinstance(material, str):
        mat = ibpy.get_material().copy()
    else:
        mat = material

    bsdf = mat.node_tree.nodes["Principled BSDF"]

    color = bsdf.inputs['Base Color'].default_value
    up = 1.2
    down = 1 / up

    if brighter > 0:
        for i in range(int(brighter)):
            color[0] *= up
            color[1] *= up
            color[2] *= up

        for i in range(3):
            color[i] = np.minimum(1, color[i])
    elif brighter < 0:
        for i in range(abs(int(brighter))):
            color[0] *= down
            color[1] *= down
            color[2] *= down

    bsdf.inputs['Base Color'].default_value = color
    return mat


def make_colorscript_bezier_curve(bob, osl_script, scale=[1, 1, 1], emission_strength=0.3):
    """
       create a material, where the coloring converts the (x,y) - position of the object into a hue-value
       color, where x,y are treated as a complex variable z=x+iy and the conversion is provided
       by an open-shader-language script


       :param emission_strength:
       :param bob:
       :param osl_script:
       :param scale:
       :return:
    """

    mat_name = str(osl_script)
    material = ibpy.get_new_material(name=mat_name)
    ibpy.create_shader_from_script(material, osl_script,
                                   scale=scale, emission_strength=emission_strength)
    # assign material
    ibpy.set_material(bob, material)


def make_voronoi_bezier_curve(bob, colors, emission_strength=0.3, scale=[1, 1, 1]):
    """
    create a material, where the coloring is a voronoi pattern

    example

    make_voronoi_bezier_curve(bob, [['white'],['white','example']])
    the first function is diplayed completely white
    the second shape of the function is turned into a voronoi pattern with alternating white and orange

    :param colors: an array of arrays
    :param bob:
    :param emission_strength:
    :param scale:
    :return:
    """
    colors = convert_strings_to_colors(colors)

    mat_name = 'voronoi' + str(colors)
    material = ibpy.get_new_material(name=mat_name)
    dialer = ibpy.create_voronoi_shader(material, colors,
                                        scale=scale, emission_strength=emission_strength)
    # assign material
    ibpy.set_material(bob, material)
    return dialer


def make_colorful_bezier_curve(bob, hue_functions, emission_strength=0.3, scale=[1, 1, 1], input='geometry_position'):
    """
    create a material, where the coloring converts the (x,y) - position of the object into a hue-value
    color, where x,y are treated as a complex variable z=x+iy and the hue_functions are the real and imaginary part of
    the transformation f(z) in reverse polish notation

    example

    make_colorful_bezier_curve(bob, ["x,x,*,y,y,*,-,1,-", "x,y,*,2,*"])
    corresponds to f(z)=z^2-1, with Re(f)=x^2-y^2-1 and Im(f)=2*x*y

    :param bob:
    :param hue_functions:
    :param emission_strength:
    :param scale:
    :return:
    """
    mat_name = str(hue_functions)
    material = ibpy.get_new_material(name=mat_name)
    dialer = ibpy.create_shader_from_function(material, hue_functions,
                                              scale=scale, emission_strength=emission_strength, input=input)
    # assign material
    ibpy.set_material(bob, material)
    return dialer


def pie_checker_material(colors=['drawing', 'joker'], name='PieChecker', **kwargs):
    """
    :param colors:
    :param name:
    :param kwargs:
    :return:
    """

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    customize_material(mat, **kwargs)

    bsdf = nodes.get('Principled BSDF')

    mixer = nodes.new(type='ShaderNodeMixRGB')
    mixer.inputs[1].default_value = get_color_from_name(colors[0])
    mixer.inputs[2].default_value = get_color_from_name(colors[1])
    mixer.location = (-200, 200)
    links.new(mixer.outputs[0], bsdf.inputs['Base Color'])
    return mat

def z_gradient(name="zGradient",**kwargs):
    """
    create a color gradient
    just a quick simple implementation, lots of customization is possible
    :param name:
    :param kwargs:
    :return:
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    mat.name = name
    tree = mat.node_tree
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    customize_material(mat, **kwargs)
    bsdf = nodes.get("Principled BSDF")

    coords = TextureCoordinate(tree, location=(-4, 0), std_out='Generated')
    sep_xyz = SeparateXYZ(tree, location=(-3, 0), vector=coords.std_out)
    ramp = ColorRamp(tree, location=(-1, 0), factor=sep_xyz.node.outputs['Z'])
    ramp.node.color_ramp.elements.new(1)
    ramp.node.color_ramp.elements[0].position = 0.00
    ramp.node.color_ramp.elements[0].color = [0, 0, 1, 1]
    ramp.node.color_ramp.elements[1].position = 0.555
    ramp.node.color_ramp.elements[1].color = [0, 1, 0, 1]
    ramp.node.color_ramp.elements[2].position = 1
    ramp.node.color_ramp.elements[2].color = [1, 0, 0, 1]
    links.new(ramp.std_out, bsdf.inputs["Base Color"])
    links.new(ramp.std_out, bsdf.inputs[EMISSION])
    return mat


def camera_gradient_rainbow(name="Rainbow",**kwargs):
    """
    create rainbow gradient across text
    This is used in video_cmb

    2024-08-25:
    The scale and the translation of the gradient are hard-coded also that the coordinate system is based on the camera.
    There should be more flexibility introduced to make it more universal
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    mat.name = name
    tree = mat.node_tree
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    customize_material(mat, **kwargs)
    bsdf = nodes.get("Principled BSDF")

    coords = TextureCoordinate(tree,location=(-4,0),std_out='Camera')
    mapping = Mapping(tree,location=(-3,0),vector=coords.std_out,loc=Vector([0.5,0,0]),scale=Vector([0.3,1,1]))
    gradient = GradientTexture(tree,location=(-2,0),gradient_type='EASING',vector=mapping.std_out,std_out=1)
    ramp = ColorRamp(tree,location=(-1,0),factor=gradient.std_out)
    ramp.node.color_ramp.elements.new(1)
    ramp.node.color_ramp.elements[0].position = 0.05
    ramp.node.color_ramp.elements[0].color = [1,0,0,1]
    ramp.node.color_ramp.elements[1].position = 0.5
    ramp.node.color_ramp.elements[1].color = [0, 1, 0, 1]
    ramp.node.color_ramp.elements[2].position = 0.95
    ramp.node.color_ramp.elements[2].color = [0, 0, 1, 1]
    links.new(ramp.std_out, bsdf.inputs["Base Color"])
    links.new(ramp.std_out, bsdf.inputs[EMISSION])
    return mat

def image_over_text(name="ImageOverText",**kwargs):
    """
    create an image texture across text
    This is used in video_cmb

    2024-08-25:
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    mat.name = name
    tree = mat.node_tree
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    customize_material(mat, **kwargs)
    bsdf = nodes.get("Principled BSDF")

    coords = TextureCoordinate(tree,location=(-4,0),std_out='Camera')
    mapping = Mapping(tree,location=(-3,0),vector=coords.std_out,loc=Vector([0.5,0,0]),scale=Vector([0.3,1,1]))
    src = get_from_kwargs(kwargs,'src',None)
    if src is not None:
        image =ibpy.get_image(src)
        image_texture = ImageTexture(tree,location=(-2,0),image=image,vector=mapping.std_out,std_out='Color')

        links.new(image_texture.std_out, bsdf.inputs["Base Color"])
        links.new(image_texture.std_out, bsdf.inputs[EMISSION])
    return mat


def polar_grid(**kwargs):
    mat = bpy.data.materials.new(name="PolarGridLines")
    mat.use_nodes = True

    mat.name = 'PolarGridLines'
    tree = mat.node_tree
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    customize_material(mat, **kwargs)
    bsdf = nodes.get("Principled BSDF")

    tex_coord = TextureCoordinate(tree, location=(-10, 0), std_out='UV')
    mapping = Mapping(tree, location=(-9, 0), vector=tex_coord.std_out)

    grid = make_function(tree, functions={
        "factor": "v_x,12,*,1,%,v_y,9,*,1,%,max"
    }, node_group_type='Shader', inputs=["v"], outputs=["factor"], vectors=["v"],
                         scalars=["factor"], location=(-8, 0))

    links.new(mapping.std_out, grid.inputs["v"])
    color_ramp = ColorRamp(tree, location=(-8, 0), factor=grid.outputs["factor"])
    color_ramp.node.color_ramp.elements[0].position = 0.990

    links.new(color_ramp.std_out, bsdf.inputs["Base Color"])
    return mat


def multipole_texture(l_max=5, **kwargs):
    """
    this texture creates a random multipole spectrum upto order l_max
    :type l_max: integer

    since the field needs to be real, we only need to consider spherical
    harmonics with non-negative m,
    and we separate them in real and imaginary parts


    """
    mat = bpy.data.materials.new(name="MultipoleTexture" + str(type))
    mat.use_nodes = True

    mat.name = 'MultipoleTexture'
    tree = mat.node_tree
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # rescale uv coordinates into (theta,phi)
    left = -10

    functions = {
        "ct": "uv_y,pi,*,cos",
        "st": "uv_y,pi,*,sin",
    }
    outs = ["ct", "st"]
    for l in range(1,l_max+1):
        functions["s"+str(l)+"f"]="uv_x,pi,*,2,*,"+str(l)+",*,sin"
        functions["c"+str(l)+"f"]="uv_x,pi,*,2,*,"+str(l)+",*,cos"
        outs.append("c"+str(l)+"f")
        outs.append("s"+str(l)+"f")



    coords = TextureCoordinate(tree, location=(left, 0))
    left += 1
    polar = make_function(nodes, functions=functions,
                          inputs=["uv"], vectors=["uv"],
                          outputs=outs,
                          scalars=outs,
                          node_group_type='ShaderNodes',
                          location=(left, -1))
    links.new(coords.std_out, polar.inputs["uv"])

    left+=1
    # setup functions with random coefficients so far

    parts = []
    last_order=None
    last_add=None
    for l in range(1,l_max+1):
        for m in range(0,l+1):
            coefficients = [np.round(10*(random()*2-1))/10 for i in range(2*(l+1))]

            y_lm = SphericalHarmonics(l,m,"theta","phi")
            parts.append(re(y_lm.poly))
            if m!=0: # for m=0 there is no imaginary part
                parts.append(im(y_lm.poly))

        # create function string
        summands = [ str(coeff)+","+ExpressionConverter(expr).postfix()+",*" for coeff, expr in zip(coefficients,parts)]

        if len(summands)>1:
            term = summands[0]+","+summands[1]+",+"
        for i in range(2,len(summands)):
            term+= ","+summands[i]+",+"
        term=term.replace("theta,cos","ct")
        term=term.replace("theta,sin","st")
        term = term.replace("phi,cos", "c1f")
        term = term.replace("phi,sin", "s1f")
        for t in range(2,l+1):
            term=term.replace(str(t)+",phi,*,cos","c"+str(l)+"f")
            term=term.replace(str(t)+",phi,*,sin","s"+str(l)+"f")
        print(term)
        order_l = make_function(nodes,functions={
            "temperature":term,
        },
                                inputs=outs[0:2*l+2],
                                outputs=["temperature"],
                                scalars = outs[0:2*l+2]+["temperature"],
                                name="order_"+str(l),
                                node_group_type="Shader",location=(left,-l))
        for label in outs[0:2*l+2]:
            links.new(polar.outputs[label],order_l.inputs[label])

        if last_order:
            if last_add:
                add = MathNode(tree,location=(left+1,-l+0.5),input0=order_l.outputs["temperature"],input1=last_add.std_out)
            else:
                add = MathNode(tree,location=(left+1,-l+0.5),input0=order_l.outputs["temperature"],input1=last_order.outputs["temperature"])
            last_add=add
        last_order = order_l

    left+=2

    abs_temp = make_function(nodes, functions={
        "abs": "temp,abs",
        "positive": "temp,0,>"
    }, inputs=["temp"], outputs=["abs", "positive"], scalars=["temp", "abs", "positive"],
                             location=(left, 0), node_group_type='ShaderNodes')
    links.new(last_add.std_out, abs_temp.inputs["temp"])
    left += 1

    # positive branch

    # color ramp for the nice color gradient
    ramp_pos = ColorRamp(tree, location=(left, 1), factor=abs_temp.outputs["abs"],
                         values=[0, 0.5, 1], colors=[[0, 1, 0, 1], [1, 1, 0, 1], [1, 0, 0, 1]], hide=False)

    # negative branch
    ramp_neg = ColorRamp(tree, location=(left, -1), factor=abs_temp.outputs["abs"],
                         values=[0, 0.5, 1],
                         colors=[[0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1]], hide=False)

    left += 2

    mix = MixRGB(tree, location=(left, 0), factor=abs_temp.outputs["positive"],
                 color1=ramp_pos.std_out, color2=ramp_neg.std_out)

    bsdf = nodes.get('Principled BSDF')
    links.new(mix.std_out, bsdf.inputs['Base Color'])
    links.new(mix.std_out, bsdf.inputs[EMISSION])

    customize_material(mat, **kwargs)
    return mat


def dipole_texture(**kwargs):
    mat = bpy.data.materials.new(name="DipoleTexture" + str(type))
    mat.use_nodes = True

    mat.name = 'DipoleTexture'
    tree = mat.node_tree
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    y11 = SphericalHarmonics(1, 1, "theta", "phi")
    y10 = SphericalHarmonics(1, 0, "theta", "phi")
    y1m1 = SphericalHarmonics(1, -1, "theta", "phi")

    a = Symbol("a", real=True)
    alpha = Symbol("alpha", real=True)
    b = Symbol("b", real=True)

    dipole = ((a + 1j * alpha) * y11.poly + b * y10.poly - (a - 1j * alpha) * y1m1.poly).expand(func=True)
    print(re(dipole))
    print("This should be zero for consistency: ", im(dipole))

    left = -7

    coords = TextureCoordinate(tree, location=(left, 0))
    left += 1
    polar = make_function(nodes, functions={
        "theta": "uv_y,pi,*",
        "phi": "uv_x,pi,*,2,*"
    },
                          inputs=["uv"], vectors=["uv"],
                          outputs=["theta", "phi"],
                          scalars=["theta", "phi"],
                          node_group_type='ShaderNodes',
                          location=(left, -1))
    links.new(coords.std_out, polar.inputs["uv"])

    a = InputValue(tree, location=(left, 1.5), value=1)
    alpha = InputValue(tree, location=(left, 1), value=-1)
    b = InputValue(tree, location=(left, 0.5), value=-1)
    left += 1

    in_sockets = [a.std_out, alpha.std_out, b.std_out, polar.outputs["theta"], polar.outputs["phi"]]
    expr = ExpressionConverter(str(re(dipole))).postfix()
    ins = ["a", "alpha", "b", "theta", "phi"]
    outs = ["temp"]
    temperature = make_function(nodes, functions={
        "temp": expr
    }, inputs=ins, outputs=outs, scalars=ins + outs,
                                node_group_type='ShaderNodes', location=(left, 0))

    for socket, label in zip(in_sockets, ins):
        links.new(socket, temperature.inputs[label])
    left += 1

    abs_temp = make_function(nodes, functions={
        "abs": "temp,abs",
        "positive": "temp,0,>"
    }, inputs=["temp"], outputs=["abs", "positive"], scalars=["temp", "abs", "positive"],
                             location=(left, 0), node_group_type='ShaderNodes')
    links.new(temperature.outputs["temp"], abs_temp.inputs["temp"])
    left += 1

    # positive branch

    # color ramp for the nice color gradient
    ramp_pos = ColorRamp(tree, location=(left, 1), factor=abs_temp.outputs["abs"],
                         values=[0, 0.5, 1], colors=[[0, 1, 0, 1], [1, 1, 0, 1], [1, 0, 0, 1]], hide=False)

    # negative branch
    ramp_neg = ColorRamp(tree, location=(left, -1), factor=abs_temp.outputs["abs"],
                         values=[0, 0.5, 1],
                         colors=[[0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1]], hide=False)

    left += 1

    mix = MixRGB(tree, location=(left, 0), factor=abs_temp.outputs["positive"],
                 color1=ramp_pos.std_out, color2=ramp_neg.std_out)

    bsdf = nodes.get('Principled BSDF')
    links.new(mix.std_out, bsdf.inputs['Base Color'])
    links.new(mix.std_out, bsdf.inputs[EMISSION])

    customize_material(mat, **kwargs)
    return mat


def rgb_color(rgb=[1, 1, 1, 1], **kwargs):
    if rgb is None:
        rgb = [1, 1, 1, 1]
    mat = bpy.data.materials.new(name="RGB")
    mat.use_nodes = True
    mat.name = 'RGB_' + str(rgb)
    nodes = mat.node_tree.nodes

    customize_material(mat, **kwargs)
    bsdf = nodes.get("Principled BSDF")
    color = rgb
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs[EMISSION].default_value = color
    return mat

def decay_mode_material(**kwargs):
    mat = bpy.data.materials.new(name="DecayModeColor" + str(type))
    mat.use_nodes = True

    mat.name = 'DecayModeColor'
    tree = mat.node_tree
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    decay_mode_attribute = AttributeNode(tree, attribute_name="DecayMode", location=(-5, 0))


    colors = [
        [0.4, 0.4, 0.4, 1],# stable
        [1, 1, 0, 1], # alpha
        [1, 0, 1, 1],# beta-
        [0, 1, 1, 1],# beta+
        [0, 0.7, 0.7, 1],# electron capture only
        [0,0,1,1],# proton emission
        [1,0,0,1], # neutron emission
        [0, 1, 0, 1],# spontaneous fission
        [1, 1, 1, 1],# unknown
         ]
    old_mix = None
    for i in range(1, 9):
        compare = MathNode(tree, location=(-4, -10 + i), operation='COMPARE', input0=decay_mode_attribute.fac_out,
                           input1=i, inputs2=0.1)
        if not old_mix:
            col = colors[i - 1]
        else:
            col = old_mix.std_out
        old_mix = MixRGB(tree, location=(-3, -10 + i), factor=compare.std_out, color1=col, color2=colors[i])

    bsdf = nodes.get('Principled BSDF')
    links.new(old_mix.std_out, bsdf.inputs['Base Color'])
    links.new(old_mix.std_out, bsdf.inputs[EMISSION])

    customize_material(mat, **kwargs)
    return mat


def star_color(temp=None, type=None, **kwargs):
    mat = bpy.data.materials.new(name="StarColor" + str(type))
    mat.use_nodes = True

    if temp is None:
        temp = type2temp(type)

    color = 1 / 255 * Vector(temp2rgb[temp] + [255])
    print(type, color)
    mat.name = 'starColor' + str(type)
    nodes = mat.node_tree.nodes
    customize_material(mat, **kwargs)
    bsdf = nodes.get("Principled BSDF")

    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs[EMISSION].default_value = color
    emission = get_from_kwargs(kwargs, "emission", 0)
    bsdf.inputs['Emission Strength'].default_value = emission

    return mat


def eight_dimensional_color(u_dim=0, v_dim=1, **kwargs):
    mat = bpy.data.materials.new(name="material_for_e8")
    mat.use_nodes = True
    u = u_dim
    v = v_dim

    mat.name = 'E8Vertices' + str(u) + str(v)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    customize_material(mat, **kwargs)
    bsdf = nodes.get("Principled BSDF")

    colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1],
              [0, 0, 0, 1]]
    mixer_sockets = []

    tex = coords = nodes.new(type='ShaderNodeTexCoord')
    coords.location = (-1000, 0)
    split = nodes.new(type='ShaderNodeSeparateXYZ')
    split.location = (- 800, 0)
    links.new(tex.outputs['UV'], split.inputs[0])

    for i in range(8):
        mix = nodes.new(type='ShaderNodeMixRGB')
        mix.location = (-i * 200, -i * 200)
        mix.inputs[1].default_value = [0, 0, 0, 1]
        if i == 0:
            links.new(mix.outputs['Color'], bsdf.inputs['Base Color'])
        else:
            links.new(mix.outputs['Color'], last_mixer.inputs[1])
        mix.inputs[2].default_value = colors[i]
        mixer_sockets.append(mix.inputs[0])
        last_mixer = mix
        if u == i:
            links.new(split.outputs[0], mix.inputs[0])
        elif v == i:
            links.new(split.outputs[1], mix.inputs[0])
        else:
            mix.inputs[0].default_value = 0

    customize_material(mat, **kwargs)
    return mat


def phase2hue_material(attribute_names=None, **kwargs):
    mat = bpy.data.materials.new(name="Phase2HueMaterial")
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    customize_material(mat, **kwargs)
    bsdf = nodes.get("Principled BSDF")

    attr = AttributeNode(tree, attribute_name=attribute_names[0], location=(-5, 0))
    trafo = make_function(tree, functions={
        "out": "in,pi,+,2,pi,*,/"
    }, location=(-4, 0), name="Transformation", inputs=["in"], outputs=["out"], scalars=["in", "out"],
                          hide=True, node_group_type='Shader')
    links.new(attr.std_out, trafo.inputs["in"])

    hue = HueSaturationValueNode(tree, location=(-3, 0), hue=trafo.outputs["out"])
    links.new(hue.std_out, bsdf.inputs['Base Color'])
    links.new(hue.std_out, bsdf.inputs[EMISSION])
    return mat


def create_material_for_e8_visuals(attribute_names=None, **kwargs):
    """
    create material specific for color coding vertices of the e8 projection

    :param attribute_names:
    :param kwargs:
    :return:
    """
    mat = bpy.data.materials.new(name="material_for_e8")
    mat.use_nodes = True
    mat.name = 'E8Vertices'
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    customize_material(mat, **kwargs)
    bsdf = nodes.get("Principled BSDF")

    colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1],
              [0, 0, 0, 1]]
    mixer_sockets = []
    for i in range(8):
        mix = nodes.new(type='ShaderNodeMixRGB')
        mix.location = (-i * 200, -i * 200)
        mix.inputs[1].default_value = [0, 0, 0, 1]
        if i == 0:
            links.new(mix.outputs['Color'], bsdf.inputs['Base Color'])
        else:
            links.new(mix.outputs['Color'], last_mixer.inputs[1])
        mix.inputs[2].default_value = colors[i]
        mixer_sockets.append(mix.inputs[0])
        last_mixer = mix

    #create attribute node and pipe the result
    for i, name in enumerate(attribute_names):
        attr = nodes.new(type='ShaderNodeAttribute')
        attr.attribute_type = "INSTANCER"
        attr.attribute_name = name
        attr.location = (-3 * i * 200 - 1000, -3 * i * 200 - 300)
        split = nodes.new(type='ShaderNodeSeparateXYZ')
        split.location = (-3 * i * 200 - 800, -3 * i * 200 - 300)
        links.new(attr.outputs['Vector'], split.inputs[0])
        if i < 2:
            comps = 3
        else:
            comps = 2
        for j in range(comps):
            trafo = nodes.new(type='ShaderNodeMath')
            trafo.operation = 'MULTIPLY_ADD'
            trafo.location = (-3 * i * 200 - 600, -3 * i * 200 - j * 50)
            trafo.hide = True
            links.new(split.outputs[j], trafo.inputs[0])
            links.new(trafo.outputs[0], mixer_sockets[j + 3 * i])

    return mat


def create_material_from_geometry_attribute(bob, attr_name='', **kwargs):
    """
    This is new stuff that I don't know where this will go to (18.2.2024)

    :param bob:
    :param attr_name:
    :param kwargs:
    :return:
    """

    mat = bpy.data.materials.new(name="material_for_" + attr_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    customize_material(mat, **kwargs)

    bsdf = nodes.get('Principled BSDF')

    attribute = nodes.new(type='ShaderNodeAttribute')
    attribute.location = (-300, 0)
    links.new(attribute.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(attribute.outputs['Color'], bsdf.inputs[EMISSION])

    attribute.attribute_name = attr_name
    attribute.attribute_type = 'GEOMETRY'

    set_material(bob, mat)

    # settings for eevee
    for slot in bob.ref_obj.material_slots:
        slot.material.blend_method = 'HASHED'
        slot.material.shadow_method = 'HASHED'


def region_indicator(region_functions=['x,1,<', 'x,1,>'], colors=['text', 'background'],
                     name=None, parameters=[], scalar_parameters=[], **kwargs):
    n_decisions = len(region_functions)
    n_colors = len(colors)
    decisions = min(n_decisions, n_colors - 1)

    # pepare colors
    for i in range(len(colors)):
        if isinstance(colors[i], str):
            colors[i] = get_color_from_name(colors[i])

    if name is None:
        name = "RegionFinder_" + str(n_decisions)
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    left = 0
    width = 200
    height = 100

    bsdf = nodes.get('Principled BSDF')
    mat_node = nodes.get('Material Output')

    mixer = nodes.new(type='ShaderNodeMixRGB')
    mixer.inputs[1].default_value = colors[0]
    mixer.inputs[2].default_value = colors[1]
    mixer.location = (left, 0)
    old_mixer = mixer

    mixers = [old_mixer]

    for i in range(2, decisions + 1):
        mixer = nodes.new(type='ShaderNodeMixRGB')
        links.new(old_mixer.outputs['Color'], mixer.inputs[1])
        mixer.inputs[2].default_value = colors[i]
        mixer.location = (left + width * (i - 1), -height * (i - 1))
        old_mixer = mixer
        mixers.append(old_mixer)

    links.new(old_mixer.outputs[0], bsdf.inputs['Base Color'])
    bsdf.location = (i * width + left, 0)
    mat_node.location = ((i + 2) * width + left, 0)

    fcn_nodes = [create_group_from_vector_function(nodes, [region_functions[i]], parameters=parameters,
                                                   scalar_parameters=scalar_parameters, name='RegionFunction' + str(i))
                 for i in range(decisions)]
    for i in range(decisions):
        fcn_nodes[i].location = (left - width, i * height)
        links.new(fcn_nodes[i].outputs[0], mixers[i].inputs[0])

    # link coordiates to the region functions
    coords = nodes.new(type='ShaderNodeTexCoord')
    coords.location = (left - 3 * width, 0)
    [links.new(coords.outputs['Object'], fcn_nodes[i].inputs['In']) for i in range(decisions)]

    alpha_factor = nodes.new(type='ShaderNodeMath')
    alpha_factor.name = 'AlphaFactor'
    alpha_factor.label = 'AlphaFactor'
    alpha_factor.operation = 'MULTIPLY'
    alpha_factor.location = ((i - 1) * width, -500)
    alpha_factor.inputs[0].default_value = 0

    links.new(alpha_factor.outputs[0], bsdf.inputs['Alpha'])
    if colors[0][3] == 0:
        # alpha value
        links.new(fcn_nodes[0].outputs[0], alpha_factor.inputs[1])

    return customize_material(mat, **kwargs)


def instance_indicator_material(colors=['drawing', 'important'],
                                name='Indicator', **kwargs):
    """
    creates a color mixer
    and a function that determines the choice

    :param function: a function that determines the mixing ratio
    :param colors: the two colors
    :return:
    """

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    customize_material(mat, **kwargs)

    bsdf = nodes.get('Principled BSDF')

    mixer = nodes.new(type='ShaderNodeMixRGB')
    mixer.inputs[1].default_value = get_color_from_name(colors[0])
    mixer.inputs[2].default_value = get_color_from_name(colors[1])
    mixer.location = (-200, 200)
    links.new(mixer.outputs[0], bsdf.inputs['Base Color'])

    if 'parameters' in kwargs:
        parameters = kwargs.get('parameters')
    else:
        parameters = []

    parameter_values = []
    for parameter in parameters:
        if parameter in kwargs:
            parameter_values.append(kwargs.pop(parameter))
        else:
            parameter_values.append(Vector())

    if 'scalar_parameters' in kwargs:
        scalar_parameters = kwargs.get('scalar_parameters')
    else:
        scalar_parameters = []

    scalar_parameter_values = []
    for parameter in scalar_parameters:
        if parameter in kwargs:
            scalar_parameter_values.append(kwargs.pop(parameter))
        else:
            scalar_parameter_values.append(0)

    # create object data
    object_node = nodes.new(type='ShaderNodeObjectInfo')
    object_node.location = (-1000, 400)

    # create indicator function
    functions = kwargs.pop('functions')
    indicator = create_group_from_vector_function(nodes, functions=functions, name='indicatorFunction', **kwargs)
    indicator.location = (-400, 200)
    links.new(indicator.outputs[0], mixer.inputs[0])

    for i, parameter in enumerate(parameters):
        comb = nodes.new(type='ShaderNodeCombineXYZ')
        comb.location = (-800, i * 200)
        comb.inputs['X'].default_value = parameter_values[i][0]
        comb.inputs['Y'].default_value = parameter_values[i][1]
        comb.inputs['Z'].default_value = parameter_values[i][2]
        links.new(comb.outputs['Vector'], indicator.inputs[parameter])
    for j, scalar_parameter in enumerate(scalar_parameters):
        val = nodes.new(type='ShaderNodeValue')
        val.location = (-800, (j + i + 1) * 200)
        val.outputs[0].default_value = scalar_parameter_values[j]
        links.new(val.outputs['Value'], indicator.inputs[scalar_parameter])

    # create transformation if necessary
    if 'transformation' in kwargs:
        transformation = kwargs.pop('transformation')

        trans_params = []
        if 'trans_params' in kwargs:
            trans_params = kwargs.pop('trans_params')

        trans_param_values = []
        for parameter in trans_params:
            if parameter in kwargs:
                trans_param_values.append(kwargs.pop(parameter))
            else:
                trans_param_values.append(Vector())

        trafo = create_group_from_vector_function(nodes, functions=[transformation], parameters=trans_params, **kwargs)
        trafo.location = (-600, 200)
        links.new(trafo.outputs[0], indicator.inputs['In'])

        for i, parameter in enumerate(trans_params):
            comb = nodes.new(type='ShaderNodeCombineXYZ')
            comb.location = (-800, i * 200)
            comb.inputs['X'].default_value = trans_param_values[i][0]
            comb.inputs['Y'].default_value = trans_param_values[i][1]
            comb.inputs['Z'].default_value = trans_param_values[i][2]
            links.new(comb.outputs['Vector'], trafo.inputs[parameter])

        links.new(object_node.outputs['Location'], trafo.inputs['In'])
    else:
        # no transformation
        links.new(object_node.outputs['Location'], indicator.inputs['In'])

    return mat


def mandelbrot_indicator_material(colors=['drawing', 'important'], name='MandelBrotSetIndicator', **kwargs):
    """
    create a material that indicates the location inside or outside the Mandelbrot set

    :param function: a function that determines the mixing ratio
    :param colors: the two colors
    :return:
    """

    if 'iterations' in kwargs:
        iterations = kwargs.pop('iterations')
    else:
        iterations = 100

    if 'threshold' in kwargs:
        threshold = kwargs.pop('threshold')
    else:
        threshold = 2

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    customize_material(mat, **kwargs)

    bsdf = nodes.get('Principled BSDF')

    mixer = nodes.new(type='ShaderNodeMixRGB')
    mixer.inputs[1].default_value = get_color(colors[0])
    mixer.inputs[2].default_value = get_color(colors[1])
    mixer.location = (-200, 200)
    links.new(mixer.outputs[0], bsdf.inputs['Base Color'])
    links.new(mixer.outputs[0], bsdf.inputs['Alpha'])

    # create object data
    object_node = nodes.new(type='ShaderNodeObjectInfo')
    object_node.location = (-1200, 400)

    # create indicator function
    indicator = create_iterator_group(nodes,
                                      ["x,x,*,z,z,*,-,c_x,+,400,min,-400,max", "y", "2,x,z,*,*,c_z,+,400,min,-400,max"],
                                      ["c"], iterations=iterations, name="MandelIterator")
    indicator.location = (-800, 200)

    length = nodes.new(type='ShaderNodeVectorMath')
    length.location = (-600, 200)
    length.operation = 'LENGTH'
    links.new(indicator.outputs[0], length.inputs[0])

    less = nodes.new(type='ShaderNodeMath')
    less.location = (-400, 200)
    less.operation = 'LESS_THAN'
    less.inputs[1].default_value = threshold
    links.new(length.outputs['Value'], less.inputs[0])
    links.new(less.outputs[0], mixer.inputs[0])

    # create transformation if necessary
    if 'transformation' in kwargs:
        transformation = kwargs.pop('transformation')

        trans_params = []
        if 'trans_params' in kwargs:
            trans_params = kwargs.pop('trans_params')

        trans_param_values = []
        for parameter in trans_params:
            if parameter in kwargs:
                trans_param_values.append(kwargs.pop(parameter))
            else:
                trans_param_values.append(Vector())

        trafo = create_group_from_vector_function(nodes, functions=[transformation], parameters=trans_params,
                                                  name='transformationNode', **kwargs)
        trafo.location = (-1000, 200)
        links.new(trafo.outputs[0], indicator.inputs['In'])

        for i, parameter in enumerate(trans_params):
            comb = nodes.new(type='ShaderNodeCombineXYZ')
            comb.location = (-1200, i * 200)
            comb.inputs['X'].default_value = trans_param_values[i][0]
            comb.inputs['Y'].default_value = trans_param_values[i][1]
            comb.inputs['Z'].default_value = trans_param_values[i][2]
            links.new(comb.outputs['Vector'], trafo.inputs[parameter])

        links.new(object_node.outputs['Location'], trafo.inputs['In'])
    else:
        # no transformation
        links.new(object_node.outputs['Location'], indicator.inputs['In'])

    return mat


def make_complex_function_material(bob, functions, shape=True, name='complex_material', **kwargs):
    """
    add texture that color-codes the phases of a complex bobject
    it works best with bobs that have a customized mesh, such as PlaneWithSingularPoints
    The x and y values of zeros and poles should be provided as special_x and special_y values
    to increase the detail at the location of these points

    :param name:
    :param functions:
    :param bob:
    :param shape: a shape key is added to show the absolute value in z direction
    :return:
    """

    if 'emission' in kwargs:
        emission = kwargs.pop('emission')
    else:
        emission = 0

    # for downward compatibility, if just one function is given
    if isinstance(functions, tuple):
        functions = functions[0]
    elif isinstance(functions, list):
        pass
    else:
        functions = [functions]

    ref = bob.ref_obj

    vert_list = ref.data.vertices
    color_map_collection = ref.data.vertex_colors

    color_maps = []

    for e, f in enumerate(functions):
        color_map = color_map_collection.new(name="color_map_" + name + "_" + str(e))
        i = 0
        for poly in ref.data.polygons:  # here was faces before
            for idx in poly.loop_indices:
                loop = ref.data.loops[idx]
                v = loop.vertex_index
                # 'z' is a complex number with the x-coordinate of the vertex being the real part
                # and the y-coordinate of the vertex the imaginary part:
                z = vert_list[v].co.x + 1j * vert_list[v].co.y
                # calculate the phase for each vertex
                w = complex(f(z))
                angle = np.angle(w)
                color_map.data[i].color = phase2rgb(angle)
                i += 1
        color_maps.append(color_map)
        print("Complex material created for " + color_map.name)

    # Connecting the Vertex Color node output to the default Principled BSDF base color input
    # to see color in rendered view:

    phase_color = bpy.data.materials.new(name="Phase Color")
    phase_color.use_nodes = True
    # for nice alpha transitions in EEVEE
    phase_color.blend_method = 'HASHED'
    phase_color.shadow_method = 'HASHED'

    links = phase_color.node_tree.links
    nodes = phase_color.node_tree.nodes
    p_bsdf = nodes.get("Principled BSDF")

    mixer_nodes = []
    mixer_dialers = []
    xpos = -400
    for i, cm in enumerate(color_maps):
        vert_col = nodes.new(type='ShaderNodeVertexColor')
        vert_col.layer_name = cm.name
        vert_col.location = (xpos, 400)

        if (i + 1) < len(color_maps):
            # create a rgb mixer for the next color_layer
            mixer = nodes.new(type='ShaderNodeMixRGB')
            mixer.inputs[0].default_value = 0
            mixer_dialers.append(mixer.inputs[0])
            mixer.location = (xpos, 200)
            xpos += 200
            links.new(vert_col.outputs[0], mixer.inputs[1])
            if len(mixer_nodes) > 0:
                links.new(mixer.outputs[0], mixer_nodes[-1].inputs[2])
            mixer_nodes.append(mixer)
        else:
            if len(mixer_nodes) > 0:
                links.new(vert_col.outputs[0], mixer_nodes[-1].inputs[2])
        xpos -= 400

    # link the layer nodes and mixers to the principled bsdf
    if len(mixer_nodes) == 0:
        # only one layer
        links.new(vert_col.outputs[0], p_bsdf.inputs['Base Color'])
        if emission > 0:
            links.new(vert_col.outputs[0], p_bsdf.inputs[EMISSION])
    else:
        links.new(mixer_nodes[0].outputs[0], p_bsdf.inputs['Base Color'])
        if emission > 0:
            links.new(mixer_nodes[0].outputs[0], p_bsdf.inputs[EMISSION])

    # assign color to object
    ref = bob.ref_obj

    # set default values for a nice texture
    alpha = bob.get_from_kwargs('alpha', 0.5)
    metallic = bob.get_from_kwargs('metallic', 1)
    roughness = bob.get_from_kwargs('roughness', 0.6)
    emission_strength = bob.get_from_kwargs('emission_strength', emission)
    transmission = bob.get_from_kwargs('transmission', 0)

    ibpy.set_alpha_for_material(phase_color, alpha)
    ibpy.set_metallic_for_material(phase_color, metallic)
    ibpy.set_roughness_for_material(phase_color, roughness)
    ibpy.set_emission_strength_for_material(phase_color, emission_strength)
    ibpy.set_transmission_for_material(phase_color, transmission)

    if len(ref.data.materials) == 0:
        ref.data.materials.append(phase_color)
    else:
        ref.material_slots[0].material = phase_color

    if len(functions) > 1:

        # create shape key for vertical displacement
        old_sk = ibpy.add_shape_key(ref, 'Basis')

        for f, function in enumerate(functions):
            # define new shape key relative to the old one
            old_sk = ibpy.add_shape_key(ref, name='Profile' + str(f), previous=old_sk)
            for i in range(len(old_sk.data)):
                x, y, z = old_sk.data[i].co[:]
                if shape:
                    w = complex(function(x + 1j * y))
                else:
                    w = 0
                old_sk.data[i].co[2] = np.abs(w)

    return mixer_dialers


def make_transformations_and_complex_material(bob, transformations, name='complex_material'):
    """
    allows for arbitrary vertex transformations
    the directions are color-coded with complex phase colors
    :param bob:
    :param transformations:
    :param name:
    :return:
    """
    ref = bob.ref_obj
    vert_list = ref.data.vertices
    color_map_collection = ref.data.vertex_colors
    color_maps = []

    # add the identity transformation as the base transformation
    all_trafos = [lambda x: x] + transformations

    for e, trafo in enumerate(all_trafos):
        color_map = color_map_collection.new(name="color_map_" + name + "_" + str(e))
        i = 0

        # count the number of polygons, each vertex belongs to.
        # vertices that only belong to two polygons are boundary polygons which are highlighted in white later on
        counter = [0 for v in range(len(vert_list))]
        vertex_color_map_data_dic = {}
        for v in range(len(vert_list)):
            vertex_color_map_data_dic[v] = []

        for poly in ref.data.polygons:
            for idx in poly.loop_indices:
                loop = ref.data.loops[idx]
                v = loop.vertex_index
                counter[v] += 1
                # 'z' is a complex number with the x-coordinate of the vertex being the real part
                # and the y-coordinate of the vertex the imaginary part:
                image = trafo(vert_list[v].co)
                z = image.x + 1j * image.y
                # calculate the phase for each vertex
                angle = np.angle(z)
                color_map.data[i].color = phase2rgb(angle)
                vertex_color_map_data_dic[v].append(i)  # collect all color_map.data_points that reference that vertex
                i += 1
        color_maps.append(color_map)

        for i, c in enumerate(counter):
            if c == 2:
                color_indices = vertex_color_map_data_dic[i]
                for c in color_indices:
                    color_map.data[c].color = [1, 1, 1, 1]

        print("Complex material created for " + color_map.name)

    # Connecting the Vertex Color node output to the default Principled BSDF base color input
    # to see color in rendered view:

    phase_color = bpy.data.materials.new(name="Phase Color")
    phase_color.use_nodes = True
    links = phase_color.node_tree.links
    nodes = phase_color.node_tree.nodes
    p_bsdf = nodes.get("Principled BSDF")

    mixer_nodes = []
    mixer_dialers = []
    xpos = -400
    for i, cm in enumerate(color_maps):
        vert_col = nodes.new(type='ShaderNodeVertexColor')
        vert_col.layer_name = cm.name
        vert_col.location = (xpos, 400)

        if (i + 1) < len(color_maps):
            # create a rgb mixer for the next color_layer
            mixer = nodes.new(type='ShaderNodeMixRGB')
            mixer.inputs[0].default_value = 0
            mixer_dialers.append(mixer.inputs[0])
            mixer.location = (xpos, 200)
            xpos += 200
            links.new(vert_col.outputs[0], mixer.inputs[1])
            if len(mixer_nodes) > 0:
                links.new(mixer.outputs[0], mixer_nodes[-1].inputs[2])
            mixer_nodes.append(mixer)
        else:
            if len(mixer_nodes) > 0:
                links.new(vert_col.outputs[0], mixer_nodes[-1].inputs[2])
        xpos -= 400

    # link the layer nodes and mixers to the principled bsdf
    if len(mixer_nodes) == 0:
        # only one layer
        links.new(vert_col.outputs[0], p_bsdf.inputs['Base Color'])
    else:
        links.new(mixer_nodes[0].outputs[0], p_bsdf.inputs['Base Color'])

    # assign color to object
    ref = bob.ref_obj

    # set default values for a nice texture
    alpha = bob.get_from_kwargs('alpha', 0.5)
    metallic = bob.get_from_kwargs('metallic', 1)
    roughness = bob.get_from_kwargs('roughness', 0.6)
    emission_strength = bob.get_from_kwargs('emission_strength', 0.2)
    transmission = bob.get_from_kwargs('transmission', 0)

    ibpy.set_alpha_for_material(phase_color, alpha)
    ibpy.set_metallic_for_material(phase_color, metallic)
    ibpy.set_roughness_for_material(phase_color, roughness)
    ibpy.set_emission_strength_for_material(phase_color, emission_strength)
    ibpy.set_transmission_for_material(phase_color, transmission)

    if len(ref.data.materials) == 0:
        ref.data.materials.append(phase_color)
    else:
        ref.material_slots[0].material = phase_color

    # create shape key for vertical displacement
    old_sk = ibpy.add_shape_key(ref, 'Basis')

    for f, function in enumerate(transformations):
        # define new shape key relative to the old one
        old_sk = ibpy.add_shape_key(ref, name='Profile' + str(f), previous=old_sk)
        for i in range(len(old_sk.data)):
            image = function(old_sk.data[i].co)
            old_sk.data[i].co = image

    return mixer_dialers


def make_conformal_transformation_material(bob, conformal_transformations, name='complex_material'):
    """
    add texture that color-codes the phases of a complex bobject
    it works best with bobs that have a customized mesh, such as PlaneWithSingularPoints
    The x and y values of zeros and poles should be provided as special_x and special_y values
    to increase the detail at the location of these points

    :param conformal_transformations: complex functions which transform the appearance of the shape
    :param name:
    :param bob:
    :param shape: a shape key is added to show the absolute value in z direction
    :return:
    """

    ref = bob.ref_obj
    vert_list = ref.data.vertices
    color_map_collection = ref.data.vertex_colors
    color_maps = []

    for e, f in enumerate(conformal_transformations):
        color_map = color_map_collection.new(name="color_map_" + name + "_" + str(e))
        i = 0
        for poly in ref.data.polygons:
            for idx in poly.loop_indices:
                loop = ref.data.loops[idx]
                v = loop.vertex_index
                # 'z' is a complex number with the x-coordinate of the vertex being the real part
                # and the y-coordinate of the vertex the imaginary part:
                z = vert_list[v].co.x + 1j * vert_list[v].co.y
                # calculate the phase for each vertex
                w = complex(f(z))
                angle = np.angle(w)
                color_map.data[i].color = phase2rgb(angle)
                i += 1
        color_maps.append(color_map)
        print("Complex material created for " + color_map.name)

    # Connecting the Vertex Color node output to the default Principled BSDF base color input
    # to see color in rendered view:

    phase_color = bpy.data.materials.new(name="Phase Color")
    phase_color.use_nodes = True
    links = phase_color.node_tree.links
    nodes = phase_color.node_tree.nodes
    p_bsdf = nodes.get("Principled BSDF")

    mixer_nodes = []
    mixer_dialers = []
    xpos = -400
    for i, cm in enumerate(color_maps):
        vert_col = nodes.new(type='ShaderNodeVertexColor')
        vert_col.layer_name = cm.name
        vert_col.location = (xpos, 400)

        if (i + 1) < len(color_maps):
            # create a rgb mixer for the next color_layer
            mixer = nodes.new(type='ShaderNodeMixRGB')
            mixer.inputs[0].default_value = 0
            mixer_dialers.append(mixer.inputs[0])
            mixer.location = (xpos, 200)
            xpos += 200
            links.new(vert_col.outputs[0], mixer.inputs[1])
            if len(mixer_nodes) > 0:
                links.new(mixer.outputs[0], mixer_nodes[-1].inputs[2])
            mixer_nodes.append(mixer)
        else:
            if len(mixer_nodes) > 0:
                links.new(vert_col.outputs[0], mixer_nodes[-1].inputs[2])
        xpos -= 400

    # link the layer nodes and mixers to the principled bsdf
    if len(mixer_nodes) == 0:
        # only one layer
        links.new(vert_col.outputs[0], p_bsdf.inputs['Base Color'])
    else:
        links.new(mixer_nodes[0].outputs[0], p_bsdf.inputs['Base Color'])

    # assign color to object
    ref = bob.ref_obj

    # set default values for a nice texture
    alpha = bob.get_from_kwargs('alpha', 0.5)
    metallic = bob.get_from_kwargs('metallic', 1)
    roughness = bob.get_from_kwargs('roughness', 0.6)
    emission_strength = bob.get_from_kwargs('emission_strength', 0.2)
    transmission = bob.get_from_kwargs('transmission', 0)

    ibpy.set_alpha_for_material(phase_color, alpha)
    ibpy.set_metallic_for_material(phase_color, metallic)
    ibpy.set_roughness_for_material(phase_color, roughness)
    ibpy.set_emission_strength_for_material(phase_color, emission_strength)
    ibpy.set_transmission_for_material(phase_color, transmission)

    if len(ref.data.materials) == 0:
        ref.data.materials.append(phase_color)
    else:
        ref.material_slots[0].material = phase_color

    # create shape key for vertical displacement
    old_sk = ibpy.add_shape_key(ref, 'Basis')

    for f, function in enumerate(conformal_transformations):
        # define new shape key relative to the old one
        old_sk = ibpy.add_shape_key(ref, name='Profile' + str(f), previous=old_sk)
        for i in range(len(old_sk.data)):
            x, y, z = old_sk.data[i].co[:]
            w = complex(function(x + 1j * y))
            old_sk.data[i].co[0] = np.real(w)
            old_sk.data[i].co[1] = np.imag(w)

    return mixer_dialers


def make_solid_material(bob, color_index):
    mat_name = "solid_" + str(color_index + 1)

    if mat_name in bpy.data.materials:
        material = bpy.data.materials[mat_name]
    else:
        # create color
        material = bpy.data.materials.new(name=mat_name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes['Principled BSDF'].inputs['Base Color'].default_value = color2rgb(COLORS[color_index])

    # assign color to object
    ref = bob.ref_obj
    if len(ref.data.materials) == 0:
        ref.data.materials.append(material)
    else:
        ref.material_slots[0].material = material


def make_glossy_material(bob, color_index):
    mat_name = "glossy_" + str(color_index + 1)

    if mat_name in bpy.data.materials:
        material = bpy.data.materials[mat_name]
    else:
        # create color
        material = bpy.data.materials.new(name=mat_name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        clear_material(material)
        nodes.new(type='ShaderNodeOutputMaterial')
        nodes.new(type='ShaderNodeBsdfGlossy')

        nodes['Glossy BSDF'].inputs['Color'].default_value = color2rgb(COLORS[color_index])
        nodes['Glossy BSDF'].inputs['Roughness'].default_value = 0.36

        material.node_tree.links.new(nodes['Glossy BSDF'].outputs['BSDF'], nodes['Material Output'].inputs[0])

    # assign color to object
    ref = bob.ref_obj
    if len(ref.data.materials) == 0:
        ref.data.materials.append(material)
    else:
        ref.material_slots[0].material = material


def make_basic_material(rgb=None, name=None):
    if rgb is None or name is None:
        raise Warning('Need rgb and name to make basic color')
    for i in range(3):
        # Range exactly 3 so a fourth component (alpha) isn't affected
        rgb[i] /= 255

    color = bpy.data.materials.new(name=name)
    color.use_nodes = True
    nodes = color.node_tree.nodes
    nodes['Principled BSDF'].inputs['Base Color'].default_value = rgb
    color.diffuse_color = rgb


def make_fake_glass_material(rgb=None, name=None, absorption_density=0.5, ior=1):
    if rgb is None or name is None:
        raise Warning('Need rgb and name to make basic color')
    for i in range(3):
        # Range exactly 3 so a fourth component (alpha) isn't affected
        rgb[i] /= 255

    color = bpy.data.materials.new(name=name)
    color.use_nodes = True
    color.use_screen_refraction = True
    links = color.node_tree.links
    nodes = color.node_tree.nodes
    bsdf = nodes['Principled BSDF']
    out = nodes.get("Material Output")

    # # frenel and transparent bsdf
    #
    # mix_shader = nodes.new(type='ShaderNodeMixShader')
    # mix_shader.location = (400, 400)
    # links.new(mix_shader.outputs['Shader'], out.inputs['Surface'])
    # links.new(bsdf.outputs['BSDF'], mix_shader.inputs[2])
    #
    # trans_bsdf = nodes.new(type='ShaderNodeBsdfTransparent')
    # trans_bsdf.location = (200, 400)
    # links.new(trans_bsdf.outputs['BSDF'], mix_shader.inputs[1])
    #
    # fresnel = nodes.new(type='ShaderNodeFresnel')
    # fresnel.location = (200, 600)
    # fresnel.inputs['IOR'].default_value = 1.3
    # links.new(fresnel.outputs['Fac'], mix_shader.inputs['Fac'])

    bsdf.inputs['Base Color'].default_value = [1, 1, 1, 1]
    bsdf.inputs['Roughness'].default_value = 0.01
    bsdf.inputs['Metallic'].default_value = 0
    bsdf.inputs[TRANSMISSION].default_value = 1
    bsdf.inputs['IOR'].default_value = ior  # no refraction
    #bsdf.inputs['Sheen Tint'].default_value = [1, 1, 1, 1]

    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    volume_absorption = nodes.new(type='ShaderNodeVolumeAbsorption')
    volume_absorption.location = (200, 200)
    volume_absorption.inputs['Color'].default_value = rgb
    volume_absorption.inputs['Density'].default_value = absorption_density
    links.new(volume_absorption.outputs['Volume'], out.inputs['Volume'])

    out.location = (400, 0)
    bsdf.location = (-400, 0)
    color.diffuse_color = rgb


def make_plastic_material(rgb=None, name=None):
    if rgb is None or name is None:
        raise Warning('Need rgb and name to make basic color')
    for i in range(3):
        # Range exactly 3 so a fourth component (alpha) isn't affected
        rgb[i] /= 255

    color = bpy.data.materials.new(name=name)
    color.use_nodes = True
    nodes = color.node_tree.nodes
    bsdf = nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = rgb
    # bsdf.inputs['Subsurface Color'].default_value=rgb
    bsdf.inputs[SPECULAR].default_value = 1
    bsdf.inputs['Roughness'].default_value = 0.1
    # bsdf.inputs['Specular Tint'].default_value=0.5


def make_checker_material():
    color = bpy.data.materials.new(name='checker')
    color.use_nodes = True
    nodes = color.node_tree.nodes
    input = nodes.new('ShaderNodeNewGeometry')
    checker = nodes.new('ShaderNodeTexChecker')
    checker.inputs['Color1'].default_value = [0.57, 0.57, 0.57, 1]
    checker.inputs['Color2'].default_value = [0.27, 0.27, 0.27, 1]
    checker.inputs['Scale'].default_value = 0.2
    color.node_tree.links.new(checker.outputs['Color'], nodes['Principled BSDF'].inputs['Base Color'])
    color.node_tree.links.new(input.outputs['Position'], checker.inputs['Vector'])


def make_mirror_material():
    color = bpy.data.materials.new(name='mirror')
    color.use_nodes = True
    nodes = color.node_tree.nodes
    bsdf = nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = [0.8, 0.8, 1, 1]
    bsdf.inputs[SPECULAR].default_value = 0
    bsdf.inputs['Metallic'].default_value = 1
    bsdf.inputs['Roughness'].default_value = 0
    bsdf.inputs['IOR'].default_value = 0


def make_magnet_material():
    for name in {"maget", "magnetX", "magnetY"}:
        material = bpy.data.materials.new(name=name)
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
        if name == "magnet":
            links.new(sep_xyz.outputs['Y'], ramp.inputs['Fac'])
            links.new(sep_xyz.outputs['Y'], ramp2.inputs['Fac'])
        if name == 'magnetY':
            links.new(sep_xyz.outputs['Y'], ramp.inputs['Fac'])
            links.new(sep_xyz.outputs['Y'], ramp2.inputs['Fac'])
        else:
            links.new(sep_xyz.outputs['X'], ramp.inputs['Fac'])
            links.new(sep_xyz.outputs['X'], ramp2.inputs['Fac'])
        coords = nodes.new(type='ShaderNodeTexCoord')
        coords.location = (-1200, 0)
        links.new(coords.outputs['Generated'], sep_xyz.inputs['Vector'])


def make_sign_material():
    material = bpy.data.materials.new(name="sign")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")

    # color ramp
    ramp = nodes.new(type='ShaderNodeValToRGB')
    ramp.location = (-300, 0)
    ramp.color_ramp.elements[0].position = 0.2
    ramp.color_ramp.elements[0].color = [1, 0, 0, 1]
    ramp.color_ramp.elements[1].position = 0.5
    ramp.color_ramp.elements[1].color = [1, 1, 1, 1]
    ramp.color_ramp.elements.new(1)
    ramp.color_ramp.elements[2].position = 0.8
    ramp.color_ramp.elements[2].color = [0, 1, 0, 1]
    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(ramp.outputs['Color'], bsdf.inputs[EMISSION])

    # map ramp
    range = nodes.new(type='ShaderNodeMapRange')
    range.location = (-500, 0)
    range.inputs['From Min'].default_value = -1
    range.inputs['From Max'].default_value = 1
    range.inputs['To Min'].default_value = 0
    range.inputs['To Max'].default_value = 1
    links.new(range.outputs['Result'], ramp.inputs['Fac'])

    # value node and two ramps
    val = nodes.new(type='ShaderNodeValue')
    val.outputs['Value'].default_value = 0
    val.location = (-700, 0)
    links.new(val.outputs['Value'], range.inputs['Value'])


def make_sand_material():
    color = bpy.data.materials.new(name='sand')
    color.use_nodes = True
    nodes = color.node_tree.nodes
    links = color.node_tree.links
    bsdf = nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = [0.80, 0.75, 0.46, 1]

    bump = nodes.new(type='ShaderNodeBump')
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    mixer = nodes.new(type='ShaderNodeMixRGB')
    mixer.inputs['Fac'].default_value = 0.90
    links.new(mixer.outputs['Color'], bsdf.inputs['Roughness'])
    links.new(mixer.outputs['Color'], bump.inputs['Height'])

    wave = nodes.new(type='ShaderNodeTexWave')
    wave.inputs['Scale'].default_value = 9.1
    wave.inputs['Distortion'].default_value = -9.9
    wave.inputs['Detail'].default_value = 2
    wave.inputs['Detail Roughness'].default_value = 0.5
    links.new(wave.outputs['Fac'], mixer.inputs['Color1'])

    noise = nodes.new(type='ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 150
    noise.inputs['Detail'].default_value = 14
    noise.inputs['Roughness'].default_value = 0.80
    links.new(noise.outputs['Fac'], mixer.inputs['Color2'])

    coords = nodes.new(type='ShaderNodeTexCoord')
    links.new(coords.outputs['Generated'], wave.inputs['Vector'])
    links.new(coords.outputs['Generated'], noise.inputs['Vector'])


def make_silk_material():
    color = bpy.data.materials.new(name='silk')
    color.use_nodes = True
    nodes = color.node_tree.nodes
    links = color.node_tree.links
    bsdf = nodes['Principled BSDF']
    bsdf.inputs['Metallic'].default_value = 1
    bsdf.inputs['Base Color'].default_value = [0.941, 0.728, 0.591, 1]
    bsdf.inputs[EMISSION].default_value = [0.941, 0.728, 0.591, 1]
    bsdf.inputs['Emission Strength'].default_value = 0.1

    bump = nodes.new(type='ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.1
    bump.inputs['Distance'].default_value = 0.1
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    magic = nodes.new(type='ShaderNodeTexMagic')
    magic.inputs['Scale'].default_value = 400
    magic.inputs['Distortion'].default_value = 0.3
    links.new(magic.outputs['Color'], bump.inputs['Height'])

    mapping = nodes.new(type='ShaderNodeMapping')
    mapping.inputs['Rotation'].default_value[2] = np.pi / 180 * 41
    links.new(mapping.outputs['Vector'], magic.inputs['Vector'])

    coords = nodes.new(type='ShaderNodeTexCoord')
    links.new(coords.outputs['UV'], mapping.inputs['Vector'])


def make_gold_material():
    color = bpy.data.materials.new(name='gold')
    color.use_nodes = True
    nodes = color.node_tree.nodes
    links = color.node_tree.links
    bsdf = nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = [0.80, 0.75, 0.46, 1]
    bsdf.inputs['Metallic'].default_value = 1
    bsdf.inputs['Roughness'].default_value = 0.27

    bump = nodes.new(type='ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.2
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    ramp = nodes.new(type='ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0
    ramp.color_ramp.elements[0].color = [1, 0.61, 0.04, 1]
    ramp.color_ramp.elements[1].position = 1
    ramp.color_ramp.elements[1].color = [1, 0.34, 0.07, 1]

    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(ramp.outputs['Color'], bump.inputs['Height'])

    noise = nodes.new(type='ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 5
    noise.inputs['Detail'].default_value = 16
    links.new(noise.outputs['Fac'], bump.inputs['Height'])

    noise2 = nodes.new(type='ShaderNodeTexNoise')
    noise2.noise_type = 'HETERO_TERRAIN'
    noise2.inputs['Scale'].default_value = 400
    noise2.inputs['Detail'].default_value = 16
    noise2.inputs['Lacunarity'].default_value = 2
    links.new(noise2.outputs[0], ramp.inputs['Fac'])

    coords = nodes.new(type='ShaderNodeTexCoord')
    links.new(coords.outputs['Object'], noise.inputs['Vector'])
    links.new(coords.outputs['Object'], noise2.inputs['Vector'])


def make_screen_material():
    color = bpy.data.materials.new(name='screen')
    color.use_nodes = True
    nodes = color.node_tree.nodes
    links = color.node_tree.links
    bsdf = nodes['Principled BSDF']
    bsdf.inputs['Metallic'].default_value = 0.9
    bsdf.inputs['Roughness'].default_value = 0.3
    bsdf.inputs['Emission Strength'].default_value = 1

    mixer = nodes.new(type='ShaderNodeMixRGB')
    mixer.location = (-200, 0)
    mixer.inputs['Color1'].default_value = [0, 0, 0, 1]
    links.new(mixer.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(mixer.outputs['Color'], bsdf.inputs[EMISSION])

    movie = nodes.new(type='ShaderNodeTexImage')
    movie.location = (-400, 0)
    links.new(movie.outputs['Color'], mixer.inputs['Color2'])
    movie.projection = 'BOX'
    movie.image_user.frame_duration = 621

    coords = nodes.new(type='ShaderNodeTexCoord')
    coords.location = (-600, 0)
    links.new(coords.outputs['Generated'], movie.inputs['Vector'])


def make_silver_material():
    color = bpy.data.materials.new(name='silver')
    color.use_nodes = True
    nodes = color.node_tree.nodes
    links = color.node_tree.links
    bsdf = nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = [0.80, 0.80, 0.80, 1]
    bsdf.inputs['Metallic'].default_value = 1
    bsdf.inputs['Roughness'].default_value = 0.

    bump = nodes.new(type='ShaderNodeBump')
    bump.location = (-300, -300)
    bump.inputs['Strength'].default_value = 0.001
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    ramp = nodes.new(type='ShaderNodeValToRGB')
    ramp.location = (-300, 300)
    ramp.color_ramp.elements[0].position = 0
    ramp.color_ramp.elements[0].color = [0.61, 0.61, 0.61, 1]
    ramp.color_ramp.elements[1].position = 1
    ramp.color_ramp.elements[1].color = [0.34, 0.34, 0.34, 1]

    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(ramp.outputs['Color'], bump.inputs['Height'])

    noise = nodes.new(type='ShaderNodeTexNoise')
    noise.location = (-500, 0)
    noise.inputs['Scale'].default_value = 5
    noise.inputs['Detail'].default_value = 16
    links.new(noise.outputs['Fac'], bump.inputs['Height'])

    noise2 = nodes.new(type='ShaderNodeTexNoise')
    noise2.noise_type = 'HETERO_TERRAIN'
    noise2.location = (-500, -300)
    noise2.inputs['Scale'].default_value = 400
    noise2.inputs['Detail'].default_value = 16
    noise2.inputs['Lacunarity'].default_value = 2
    links.new(noise2.outputs[0], ramp.inputs['Fac'])

    coords = nodes.new(type='ShaderNodeTexCoord')
    coords.location = (-800, 0)
    links.new(coords.outputs['Generated'], noise.inputs['Vector'])
    links.new(coords.outputs['Generated'], noise2.inputs['Vector'])


def make_scattering_material(**kwargs):
    color = bpy.data.materials.new(name='scatter_volume')
    color.use_nodes = True
    nodes = color.node_tree.nodes
    links = color.node_tree.links
    material = nodes['Material Output']
    bsdf = nodes['Principled BSDF']
    nodes.remove(bsdf)
    # make new bsdf that is not coupled to the material output as a short hack to be able to manipulate transparency during the appearance of the object

    scatter = nodes.new(type='ShaderNodeVolumeScatter')  # index 5
    density = get_from_kwargs(kwargs, 'density', 0.05)
    anisotropy = get_from_kwargs(kwargs, 'anisotropy', 0)
    nodes['Volume Scatter'].inputs['Density'].default_value = density
    nodes['Volume Scatter'].inputs['Anisotropy'].default_value = anisotropy

    links.new(scatter.outputs['Volume'], material.inputs['Volume'])

    return color


def make_marble_material():
    color = bpy.data.materials.new(name='marble')
    color.use_nodes = True
    nodes = color.node_tree.nodes
    links = color.node_tree.links
    bsdf = nodes['Principled BSDF']
    bsdf.inputs['Roughness'].default_value = 0.15

    bump = nodes.new(type='ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.08
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    ramp = nodes.new(type='ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0.386
    ramp.color_ramp.elements[0].color = [1, 1, 1, 1]
    ramp.color_ramp.elements[1].position = 0.745
    ramp.color_ramp.elements[1].color = [0.010, 0.034, 0.047, 1]

    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(ramp.outputs['Color'], bump.inputs['Height'])

    noise = nodes.new(type='ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 5
    noise.inputs['Detail'].default_value = 16
    links.new(noise.outputs['Fac'], ramp.inputs['Fac'])

    vec_map = nodes.new(type='ShaderNodeMapping')
    vec_map.inputs[1].default_value = [0, 0, 0.9]
    links.new(vec_map.outputs['Vector'], noise.inputs['Vector'])

    noise2 = nodes.new(type='ShaderNodeTexNoise')
    noise2.inputs['Scale'].default_value = 2
    noise2.inputs['Detail'].default_value = 16
    links.new(noise2.outputs['Fac'], vec_map.inputs['Vector'])

    coords = nodes.new(type='ShaderNodeTexCoord')
    links.new(coords.outputs['Generated'], noise2.inputs['Vector'])


def make_metal_materials():
    for i in range(1, 10):
        gray = i / 10
        make_metal_material(gray=gray)


def make_metal_material(gray=0.5):
    color = bpy.data.materials.new(name='metal_' + str(gray))
    color.use_nodes = True
    nodes = color.node_tree.nodes
    bsdf = nodes['Principled BSDF']
    bsdf.inputs['Metallic'].default_value = 1
    bsdf.inputs['Roughness'].default_value = 0.1
    bsdf.inputs['Base Color'].default_value = [gray, gray, gray, 1]


def make_wood_material():
    color = bpy.data.materials.new(name='wood')

    color.use_nodes = True
    nodes = color.node_tree.nodes
    links = color.node_tree.links
    bsdf = nodes['Principled BSDF']

    bump = nodes.new(type='ShaderNodeBump')
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    ramp = nodes.new(type='ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 1
    ramp.color_ramp.elements[0].color = [1, 1, 1, 1]
    ramp.color_ramp.elements[1].position = 0.123
    ramp.color_ramp.elements[1].color = [0.3, 0.3, 0.3, 1]
    links.new(ramp.outputs['Color'], bsdf.inputs['Roughness'])

    ramp2 = nodes.new(type='ShaderNodeValToRGB')
    ramp2.color_ramp.elements[0].position = 0.295
    ramp2.color_ramp.elements[0].color = [0] * 4
    ramp2.color_ramp.elements[1].position = 0.70
    ramp2.color_ramp.elements[1].color = [1, 0.78, .62, 1]
    ramp2.color_ramp.elements.new(0.468)
    ramp2.color_ramp.elements[-1].color = [0.16, 0.05, 0.01, 1]
    ramp2.color_ramp.elements.new(0.641)
    ramp2.color_ramp.elements[-1].color = [0.52, 0.28, 0.11, 1]
    links.new(ramp2.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(ramp2.outputs['Color'], ramp.inputs['Fac'])

    voronoi = nodes.new(type='ShaderNodeTexVoronoi')
    voronoi.inputs['Scale'].default_value = 6
    voronoi.inputs['Randomness'].default_value = 1
    links.new(voronoi.outputs['Distance'], ramp2.inputs['Fac'])

    voronoi2 = nodes.new(type='ShaderNodeTexVoronoi')
    voronoi2.inputs['Scale'].default_value = 6
    voronoi2.inputs['Randomness'].default_value = 1
    links.new(voronoi2.outputs['Distance'], bump.inputs['Height'])

    noise = nodes.new(type='ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 2
    noise.inputs['Detail'].default_value = 16
    noise.inputs['Roughness'].default_value = 0.8
    links.new(noise.outputs['Fac'], voronoi.inputs['Vector'])

    noise2 = nodes.new(type='ShaderNodeTexNoise')
    noise2.inputs['Scale'].default_value = 5.4
    noise2.inputs['Detail'].default_value = 16
    noise2.inputs['Roughness'].default_value = 0.3
    links.new(noise2.outputs['Fac'], voronoi2.inputs['Vector'])

    mapping = nodes.new(type='ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = [10, 1, 1]
    links.new(mapping.outputs['Vector'], noise.inputs['Vector'])
    links.new(mapping.outputs['Vector'], noise2.inputs['Vector'])

    coords = nodes.new(type='ShaderNodeTexCoord')
    links.new(coords.outputs['Generated'], mapping.inputs['Vector'])


def make_creature_material(rgb=None, name=None):
    if rgb is None or name is None:
        raise Warning('Need rgb and name to make creature color')
    for i in range(3):
        # Range exactly 3 so a fourth component (alpha) isn't affected
        rgb[i] /= 255

    color = bpy.data.materials.new(name=name)
    color.use_nodes = True
    nodes = color.node_tree.nodes
    # nodes[1].inputs[1].default_value = 1 #Roughness. 1 means not shiny.
    nodes['Principled BSDF'].inputs['Base Color'].default_value = rgb
    color.node_tree.links.new(nodes['Principled BSDF'].outputs[0], nodes['Material Output'].inputs[0])

    # which doesn't take alpha
    color.diffuse_color = rgb


def make_translucent_material(rgb=None, name=None):
    if rgb is None or name is None:
        raise Warning('Need rgb and name to make translucent color')
    for i in range(3):
        # Range exactly 3 so a fourth component (alpha) isn't affected
        rgb[i] /= 255

    strength = 0.5  # Arbitrary, could make this a constant
    # strength = 0.1

    color = bpy.data.materials.new(name=name)
    color.use_nodes = True
    nodes = color.node_tree.nodes
    links = color.node_tree.links
    material = nodes['Material Output']
    # color.node_tree.links.remove(nodes[0].inputs[0].links[0])
    bsdf = nodes['Principled BSDF']
    if bsdf:
        nodes.remove(bsdf)

    # add mixed shader to allow for fading in and out
    shader1 = nodes.new(type='ShaderNodeMixShader')
    shader2 = nodes.new(type='ShaderNodeMixShader')
    alpha = nodes.new(type='ShaderNodeBsdfTransparent')
    alpha.inputs['Color'].default_value = rgb
    links.new(alpha.outputs[0], shader1.inputs[1])
    links.new(alpha.outputs[0], shader2.inputs[1])
    shader1.inputs[0].default_value = 0  # transparent at the beginning
    shader2.inputs[0].default_value = 0  # transparent
    links.new(shader1.outputs[0], material.inputs['Surface'])
    links.new(shader2.outputs[0], material.inputs['Volume'])

    nodes.new(type='ShaderNodeAddShader')  # index 2
    color.node_tree.links.new(nodes['Add Shader'].outputs[0], shader2.inputs[2])
    nodes.new(type='ShaderNodeAddShader')  # index 3
    color.node_tree.links.new(nodes['Add Shader.001'].outputs[0], nodes['Add Shader'].inputs[1])
    nodes.new(type='ShaderNodeEmission')  # index 4
    nodes['Emission'].inputs['Color'].default_value = rgb
    nodes['Emission'].inputs['Strength'].default_value = strength
    color.node_tree.links.new(nodes['Emission'].outputs[0], nodes['Add Shader'].inputs[0])
    nodes.new(type='ShaderNodeVolumeScatter')  # index 5
    nodes['Volume Scatter'].inputs['Color'].default_value = rgb
    nodes['Volume Scatter'].inputs['Density'].default_value = strength
    color.node_tree.links.new(nodes['Volume Scatter'].outputs['Volume'], nodes['Add Shader.001'].inputs[0])
    absorption = nodes.new(type='ShaderNodeVolumeAbsorption')  # index 6
    absorption.inputs['Color'].default_value = rgb
    absorption.inputs['Density'].default_value = strength
    color.node_tree.links.new(absorption.outputs['Volume'], nodes['Add Shader.001'].inputs[1])
    glass = nodes.new(type='ShaderNodeBsdfGlass')
    glass.inputs['Color'].default_value = [1, 1, 1, 1]
    color.node_tree.links.new(glass.outputs['BSDF'], shader1.inputs[2])


def mandel_on_riemann_sphere(**kwargs):
    if 'iterations' in kwargs:
        iterations = kwargs.pop('iterations')
    else:
        iterations = 10

    if 'ramp_colors' in kwargs:
        ramp_colors = kwargs.pop('ramp_colors')
    else:
        ramp_colors = ["background", "text"]

    material = bpy.data.materials.new(name='mandel_on_riemann_sphere')

    # for eevee
    material.use_screen_refraction = True
    # for cycles
    material.cycles.displacement_method = 'DISPLACEMENT'  # for real displacement
    material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes['Principled BSDF']
    out = nodes.get('Material Output')

    outer_left = -1600

    texture_coords = nodes.new(type='ShaderNodeTexCoord')
    texture_coords.location = (outer_left - 1000, 400)

    # rescale uv -coordinates

    scaling = nodes.new(type='ShaderNodeVectorMath')
    scaling.location = (outer_left - 800, 400)
    scaling.operation = "MULTIPLY_ADD"
    scaling.inputs[1].default_value = Vector([np.pi, 2 * np.pi, 1])
    scaling.inputs[2].default_value = Vector([-np.pi, 0, 0])
    links.new(texture_coords.outputs['UV'], scaling.inputs[0])

    # discretizer
    discretizer = create_group_from_vector_function(nodes,
                                                    ["x,spacing_x,/,floor,spacing_x,*",
                                                     "y,spacing_y,/,floor,spacing_y,*", "z"],
                                                    parameters=["spacing"], name="Discretizer")

    discretizer.location = (outer_left - 400, 600)
    links.new(scaling.outputs[0], discretizer.inputs[0])
    coords = discretizer.outputs[0]

    spacing_vec = nodes.new(type='ShaderNodeCombineXYZ')
    spacing_vec.location = (outer_left - 400, 1000)

    val = nodes.new(type='ShaderNodeValue')
    val.location = (outer_left - 600, 1000)
    val.label = 'Spacing'
    val.outputs[0].default_value = 0.05

    thickness_vec = nodes.new(type='ShaderNodeCombineXYZ')
    thickness_vec.location = (outer_left - 400, 1200)

    thickness = nodes.new(type='ShaderNodeValue')
    thickness.location = (outer_left - 600, 1200)
    thickness.label = 'Thickness'
    thickness.outputs[0].default_value = 0.005
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
    links.new(scaling.outputs[0], grid.inputs[0])
    links.new(spacing_vec.outputs[0], grid.inputs[1])
    links.new(thickness_vec.outputs[0], grid.inputs[2])

    # coloring

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
    pixel_iterator.location = (-200, 1000)
    links.new(discretizer.outputs[0], pixel_iterator.inputs[0])
    links.new(spacing_vec.outputs[0], pixel_iterator.inputs[2])

    val_x = nodes.new(type='ShaderNodeValue')
    val_x.location = (-600, 1100)
    val_x.name = 'XCoordParam'
    val_x.label = 'XCoordParam'
    val_x.outputs[0].default_value = 0
    val_x.hide = True

    val_y = nodes.new(type='ShaderNodeValue')
    val_y.location = (-600, 1050)
    val_y.name = 'YCoordParam'
    val_y.label = 'YCoordParam'
    val_y.outputs[0].default_value = 0
    val_y.hide = True

    combine = nodes.new('ShaderNodeCombineXYZ')
    combine.location = (-400, 1000)
    combine.hide = True
    links.new(val_x.outputs[0], combine.inputs[0])
    links.new(val_y.outputs[0], combine.inputs[1])
    links.new(combine.outputs[0], pixel_iterator.inputs[1])

    mixer = nodes.new(type='ShaderNodeMixRGB')
    mixer.location = (-100, 0)
    mixer.inputs[1].default_value = get_color_from_string("gray_2")
    links.new(pixel_iterator.outputs[0], mixer.inputs[0])
    links.new(vec_multi.outputs[0], mixer.inputs[2])

    links.new(mixer.outputs[0], bsdf.inputs['Base Color'])
    links.new(mixer.outputs[0], bsdf.inputs[EMISSION])

    # create pixel corners
    coords_dl = coords
    coords_dr = create_group_from_vector_function(nodes, ['x,spacing_x,+', 'y,0,+'], ['spacing'], name="groupDR")
    coords_dr.location = (outer_left - 100, 400)
    coords_ul = create_group_from_vector_function(nodes, ['x,0,+', 'y,spacing_y,+'], ['spacing'], name="groupUL")
    coords_ul.location = (outer_left - 100, 200)
    coords_ur = create_group_from_vector_function(nodes, ['x,spacing_x,+', 'y,spacing_y,+'], ['spacing'],
                                                  name="groupUR")
    coords_ur.location = (outer_left - 100, 0)

    links.new(spacing_vec.outputs[0], coords_dr.inputs[1])
    links.new(spacing_vec.outputs[0], coords_ul.inputs[1])
    links.new(spacing_vec.outputs[0], coords_ur.inputs[1])
    links.new(coords, coords_dr.inputs[0])
    links.new(coords, coords_ul.inputs[0])
    links.new(coords, coords_ur.inputs[0])

    # create projector for each iterator_group
    projection_dl = create_group_from_vector_function(nodes, functions=[
        "1,x,cos,+,2,*,x,-1,*,sin,/,y,cos,*",
        "1,x,cos,+,2,*,x,-1,*,sin,/,y,sin,*",
        "z"
    ], name='stereoProjection')
    projection_dl.location = (outer_left + 50, 400)
    links.new(coords_dl, projection_dl.inputs[0])

    projection_dr = create_group_from_vector_function(nodes, functions=[
        "1,x,cos,+,2,*,x,-1,*,sin,/,y,cos,*",
        "1,x,cos,+,2,*,x,-1,*,sin,/,y,sin,*",
        "z"
    ], name='stereoProjection')
    projection_dr.location = (outer_left + 50, 200)
    links.new(coords_dr.outputs[0], projection_dr.inputs[0])

    projection_ul = create_group_from_vector_function(nodes, functions=[
        "1,x,cos,+,2,*,x,-1,*,sin,/,y,cos,*",
        "1,x,cos,+,2,*,x,-1,*,sin,/,y,sin,*",
        "z"
    ], name='stereoProjection')
    projection_ul.location = (outer_left + 50, 0)
    links.new(coords_ul.outputs[0], projection_ul.inputs[0])

    projection_ur = create_group_from_vector_function(nodes, functions=[
        "1,x,cos,+,2,*,x,-1,*,sin,/,y,cos,*",
        "1,x,cos,+,2,*,x,-1,*,sin,/,y,sin,*",
        "z"
    ], name='stereoProjection')
    projection_ur.location = (outer_left + 50, -200)
    links.new(coords_ur.outputs[0], projection_ur.inputs[0])

    # create iterator groups
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

    links.new(projection_dl.outputs[0], iterator_group_dl.inputs[0])
    links.new(projection_dr.outputs[0], iterator_group_dr.inputs[0])
    links.new(projection_ul.outputs[0], iterator_group_ul.inputs[0])
    links.new(projection_ur.outputs[0], iterator_group_ur.inputs[0])

    # check, whether all for corners of the pixel stay below 2 after iterations
    and_group = create_group_from_vector_function(nodes,
                                                  ['v,length,2,<,dr,length,2,<,*,ul,length,2,<,*,ur,length,2,<,*'],
                                                  parameters=['dr', 'ul', 'ur'], name='andGroup')
    and_group.location = (outer_left + 500, 400)
    links.new(iterator_group_dl.outputs[0], and_group.inputs[0])
    links.new(iterator_group_dr.outputs[0], and_group.inputs[1])
    links.new(iterator_group_ul.outputs[0], and_group.inputs[2])
    links.new(iterator_group_ur.outputs[0], and_group.inputs[3])
    links.new(and_group.outputs[0], ramp.inputs[0])

    mul = nodes.new('ShaderNodeMath')
    mul.location = (-400, 0)
    mul.operation = 'MULTIPLY'
    mul.inputs[1].default_value = 0.5
    links.new(pixel_iterator.outputs[0], mul.inputs[0])
    links.new(and_group.outputs[0], mul.inputs[1])

    displacement = nodes.new(type='ShaderNodeDisplacement')
    displacement.location = (0, -800)
    displacement.inputs['Midlevel'].default_value = 0
    displacement.inputs['Scale'].default_value = -0.0225
    links.new(mul.outputs[0], displacement.inputs['Height'])
    links.new(displacement.outputs['Displacement'], out.inputs['Displacement'])

    return material


def coarse_graining(bob, **kwargs):
    obj = get_obj(bob)
    # select the channel that should be coarse grained

    if 'src' in kwargs:
        src = kwargs.pop('src')
    else:
        src = 'uv'

    material = obj.material_slots[0].material

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes['Principled BSDF']
    out = nodes.get('Material Output')

    # for eevee
    material.use_screen_refraction = True
    # for cycles
    material.cycles.displacement_method = 'DISPLACEMENT'  # for real displacement
    material.use_nodes = True

    outer_left = -1000

    if src == 'uv':
        texture_coords = nodes.new(type='ShaderNodeTexCoord')
        texture_coords.location = (outer_left, 400)
        coords = texture_coords.ouputs['UV']
    elif src == 'object':
        texture_coords = nodes.new(type='ShaderNodeTexCoord')
        texture_coords.location = (outer_left, 400)
        coords = texture_coords.ouputs['Object']
    elif src == 'img':
        texture_coords = nodes.new(type='ShaderNodeTexCoord')
        texture_coords.location = (outer_left, 400)
        coords = texture_coords.outputs['Generated']
        texture = nodes.get('Image Texture')

    coarse_grained = create_group_from_vector_function(nodes, functions=[
        'v,(0.5 0.5 0.5),sub,spacing,div,(0.5 0.5 0.5),add,vfloor,spacing,mul,(0.5 0.5 0.5),add'
    ], parameters=['spacing'])
    coarse_grained.location = (outer_left + 200, 400)

    spacing = nodes.new(type='ShaderNodeCombineXYZ')
    spacing.location = (outer_left, 0)
    for i in range(3):
        spacing.inputs[i].default_value = 1

    value = nodes.new(type='ShaderNodeValue')
    value.outputs[0].default_value = 0.5
    value.location = (outer_left - 200, 0)
    value.name = 'Spacing'
    value.label = 'Spacing'

    links.new(value.outputs[0], spacing.inputs[0])
    links.new(value.outputs[0], spacing.inputs[1])
    links.new(value.outputs[0], spacing.inputs[2])
    links.new(spacing.outputs[0], coarse_grained.inputs['spacing'])
    links.new(coords, coarse_grained.inputs[0])
    links.new(coarse_grained.outputs[0], texture.inputs[0])

    return material


def monte_carlo_mandel(bob, **kwargs):
    obj = get_obj(bob)

    material = obj.material_slots[0].material

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes['Principled BSDF']
    out = nodes.get('Material Output')

    # for eevee
    material.use_screen_refraction = True
    # for cycles
    material.cycles.displacement_method = 'DISPLACEMENT'  # for real displacement
    material.use_nodes = True


def penrose_material(base_color, contrast=1, **kwargs):
    material = bpy.data.materials.new(name='Penrose')
    if 'emission' in kwargs:
        emission = kwargs.pop('emission')
    else:
        emission = 0
    # for eevee
    material.use_screen_refraction = True
    # for cycles
    material.cycles.displacement_method = 'DISPLACEMENT'  # for real displacement
    material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes['Principled BSDF']
    out = nodes.get('Material Output')

    left = -1000
    length = 0
    width = 200

    # link area attribute
    attribute = nodes.new(type='ShaderNodeAttribute')
    attribute.location = (left + length * width, 0)
    attribute.attribute_name = 'Size'
    attribute.attribute_type = 'GEOMETRY'
    length += 1

    map = nodes.new(type='ShaderNodeMapRange')
    map.location = (left + length * width, 0)
    map.inputs['To Min'].default_value = 1 - contrast
    links.new(attribute.outputs['Color'], map.inputs['Value'])
    length += 1

    node_hue = nodes.new(type='ShaderNodeHueSaturation')
    node_hue.location = (left + length * width, 0)
    node_hue.inputs['Color'].default_value = get_color_from_name(base_color)
    links.new(map.outputs['Result'], node_hue.inputs['Value'])
    length += 1

    mixer = nodes.new(type='ShaderNodeMixRGB')
    mixer.inputs[0].default_value = 0
    mixer.location = (left + (length * width), 0)
    length += 1
    links.new(node_hue.outputs[0], mixer.inputs[1])

    bsdf.location = (left + length * width, 0)
    bsdf.inputs['Emission Strength'].default_value = emission
    links.new(mixer.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(mixer.outputs['Color'], bsdf.inputs[EMISSION])

    length += 2
    out.location = (left + length * width, 0)

    # prepare the mixing into the logo

    length = -3
    # link area attribute
    attribute = nodes.new(type='ShaderNodeAttribute')
    attribute.location = (left + length * width, -300)
    attribute.attribute_name = 'Center'
    attribute.attribute_type = 'GEOMETRY'
    length += 1

    div = nodes.new(type='ShaderNodeVectorMath')
    div.operation = 'DIVIDE'
    div.location = (left + length * width, -300)
    if 'scaling' in kwargs:
        scaling = kwargs.pop('scaling')
    else:
        scaling = [608, 612, 0]
    div.inputs[1].default_value = scaling
    links.new(attribute.outputs['Vector'], div.inputs[0])
    length += 1

    shift = nodes.new(type='ShaderNodeVectorMath')
    shift.operation = 'ADD'
    shift.location = (left + length * width, -300)
    shift.inputs[1].default_value = [0.5] * 3
    links.new(div.outputs['Vector'], shift.inputs[0])
    length += 1

    img = nodes.new(type='ShaderNodeTexImage')
    img.location = (left + length * width, -300)
    img.image = bpy.data.images.load(os.path.join(IMG_DIR, "logo.png"))
    img.extension = 'EXTEND'
    links.new(shift.outputs[0], img.inputs['Vector'])
    links.new(img.outputs['Color'], mixer.inputs[2])
    # settings for eevee
    material.blend_method = 'HASHED'
    material.shadow_method = 'HASHED'

    return material


def material_clean_up():
    # Function for removing some duplicate materials from repeated imports
    for mat in bpy.data.materials:
        if 'color' not in mat.name:
            bpy.data.materials.remove(mat)


def get_alpha_of_material(material):
    if isinstance(material, str):
        material = bpy.data.materials[material]
    nodes = material.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    return bsdf.inputs['Alpha']


#################
# backgrounds ###
#################


def set_sky_background(**kwargs):
    # remove lights
    for obj in bpy.data.objects:
        if 'Sun' in obj.name:
            ibpy.un_link(obj, collection='Collection')
    world = bpy.data.worlds[-1]

    nodes = world.node_tree.nodes
    links = world.node_tree.links

    out = nodes['World Output']
    out.location=(200,0)

    sky = nodes.new(type="ShaderNodeTexSky")
    sky.sky_type="NISHITA"
    altitude=get_from_kwargs(kwargs,'altitude',0)
    intensity=get_from_kwargs(kwargs,'sun_intensity',1)
    sky.location=(-200,0)
    sky.altitude=altitude
    sky.sun_intensity=intensity

    air = get_from_kwargs(kwargs, 'air_density', 1)
    dust = get_from_kwargs(kwargs, 'dust_density', 1)
    ozone = get_from_kwargs(kwargs, 'ozone_density', 1)

    sky.air_density=air
    sky.dust_density=dust
    sky.ozone_density=ozone

    background = nodes.get("Background")
    background.location=(0,0)

    links.new(sky.outputs['Color'],background.inputs['Color'])
    links.new(background.outputs['Background'],out.inputs["Surface"])

    animate_sky_background(sky,**kwargs)
