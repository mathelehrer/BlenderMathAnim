import os

import bpy
import numpy as np
from mathutils import Vector

from geometry_nodes.nodes import Frame, layout, make_function
from utils.constants import IMG_DIR
from utils.kwargs import get_from_kwargs
from utils.string_utils import parse_vector

pi = np.pi

SOCKET_TYPES = {"STRING": 'NodeSocketString', "BOOLEAN": 'NodeSocketBool', "MATERIAL": 'NodeSocketMaterial',
                "VECTOR": 'NodeSocketVector', "INT": 'NodeSocketInt', "MENU": 'NodeSocketMenu',
                "COLLECTION": 'NodeSocketCollection',
                "GEOMETRY": 'NodeSocketGeometry', "TEXTURE": 'NodeSocketTexture', "FLOAT": 'NodeSocketFloat',
                "COLOR": 'NodeSocketColor', "OBJECT": 'NodeSocketObject', "ROTATION": 'NodeSocketRotation',
                "MATRIX": 'NodeSocketMatrix', "IMAGE": 'NodeSocketImage', "VALUE": 'NodeSocketFloat'}

# auxiliary functions

def maybe_flatten(list_of_lists):
    result = []
    for part in list_of_lists:
        if isinstance(part, list):
            result += part
        else:
            result.append(part)
    return result


def parse_location(location):
    location = location.replace("(", "")
    location = location.replace(")", "")
    coords = location.split(",")
    return (float(coords[0]), float(coords[1]))


def parse_bool(bool_string):
    if bool_string == "True":
        return True
    else:
        return False


class ShaderNode:
    def __init__(self, tree, location=(0, 0), width=200, height=100, **kwargs):
        self.tree = tree
        self.location = location
        self.l, self.m = location
        self.node.location = (self.l * width, self.m * height)
        self.links = tree.links
        self.node.hide = True
        self.inputs = self.node.inputs
        self.outputs = self.node.outputs

        if 'hide' in kwargs:
            hide = kwargs.pop('hide')
            self.node.hide = hide
        if 'name' in kwargs:
            name = kwargs.pop('name')
            self.node.label = name
            self.node.name = name
            self.name = name
        if 'label' in kwargs:
            label = kwargs.pop('label')
            self.node.label = label

    @classmethod
    def from_attributes(cls, tree, attributes):
        name = attributes["name"]
        # print("Create Node from attributes: ",attributes["id"],": ", name)
        if "location" not in attributes:
            pass
        location = parse_location(attributes["location"])
        label = attributes["label"]
        if attributes["hide"] == "False":
            hide = False
        else:
            hide = True

        if attributes["mute"] == "False":
            mute = False
        else:
            mute = True

        type = attributes["type"]
        if type == "ATTRIBUTE":
            attribute_name = attributes["attribute_name"]
            attribute_type = attributes.get("attribute_type", "GEOMETRY")
            return AttributeNode(tree, location=location, label=label,
                                 name=name, attribute_name=attribute_name,
                                 attribute_type=attribute_type,
                                 hide=hide, mute=mute, node_height=200)
        elif type == "BRIGHTCONTRAST":
            return BrightContrast(tree, location=location, label=label, name=name, hide=hide, mute=mute,
                                  node_height=200)
        elif type == "BSDF_GLASS":
            distribution = attributes["distribution"]
            return BSDFGlass(tree, location=location,
                             label=label, name=name,
                             hide=hide, mute=mute,
                             distribution=distribution,
                             node_height=200)
        elif type == "BSDF_PRINCIPLED":
            return PrincipledBSDF(tree, location=location, label=label,
                                  name=name, hide=hide, mute=mute,
                                  node_height=600)
        elif type == "BSDF_TRANSPARENT":
            return BSDFTransparent(tree, location=location, label=label, name=name, hide=hide, mute=mute,
                                   node_height=200)
        elif type == "BUMP":
            invert = parse_bool(attributes["invert"])
            return Bump(tree, location=location, label=label, name=name, hide=hide, mute=mute,
                        invert=invert, node_height=200)
        elif type == "COMBXYZ":
            return CombineXYZ(tree, location=location, label=label, name=name, hide=hide, mute=mute, node_height=200)
        elif type == "DISPLACEMENT":
            space = attributes["space"]
            return Displacement(tree, location=location, label=label, name=name, hide=hide, mute=mute,
                                space=space, node_height=400)
        elif type == "FRAME":
            return Frame(tree, location=location, name=name, label=label, hide=hide, mute=mute, node_height=200,
                         node_width=200)
        elif type == "HUE_SAT":
            return HueSaturationValueNode(tree, location=location, name=name, label=label,
                                          hide=hide, mute=mute, node_height=200,
                                          node_width=200)
        elif type == "MAPPING":
            vector_type = attributes["vector_type"]
            return Mapping(tree, location, vector_type=vector_type, label=label, hide=hide, mute=mute, node_height=200,
                           node_width=200)
        elif type == "MATH":
            operation = attributes["operation"]
            return MathNode(tree, location=location, label=label, name=name, hide=hide, mute=mute,
                            operation=operation, node_height=200)
        elif type == "MIX":
            data_type = attributes["data_type"]
            return MixNode(tree, location=location, label=label, name=name, hide=hide, mute=mute,
                           data_type=data_type, node_height=200)
        elif type == "MIX_SHADER":
            return MixShader(tree, location=location, label=label, name=name, hide=hide, mute=mute, node_height=200)
        elif type == "OUTPUT_MATERIAL":
            return OutputMaterial(tree, location=location, label=label, name=name, hide=hide, mute=mute,
                                  node_height=200)
        elif type == "PRINCIPLED_VOLUME":
            return PrincipledVolume(tree, location=location, label=label,
                                    name=name, hide=hide, mute=mute,
                                    node_height=200)
        elif type == "RGB":
            return RGB(tree, location=location, label=label,
                       name=name, hide=hide, mute=mute,
                       node_height=200)
        elif type == "SEPXYZ":
            return SeparateXYZ(tree, location=location, label=label, name=name, hide=hide, mute=mute, node_height=200)
        elif type == "TEX_COORD":
            return TextureCoordinate(tree, location=location, label=label, name=name, hide=hide, mute=mute,
                                     node_height=200)
        elif type == "TEX_IMAGE":
            extension = attributes["extension"]
            image_name = attributes["image.name"]
            image_source = attributes["image.source"]
            image_colorspace_settings_name = attributes["image.colorspace_settings.name"]
            interpolation = attributes["interpolation"]
            projection = attributes["projection"]
            return TextureImage(tree, location=location, label=label, name=name, hide=hide, mute=mute,
                                extension=extension, image_name=image_name, image_source=image_source,
                                image_colorspace_settings_name=image_colorspace_settings_name,
                                interpolation=interpolation, projection=projection)
        elif type == "TEX_NOISE":
            noise_dimensions = attributes["noise_dimensions"]
            noise_type = attributes["noise_type"]
            bool_str = attributes["normalize"]
            if bool_str == "True":
                normalize = True
            else:
                normalize = False
            return NoiseTexture(tree, location=location, label=label, name=name, hide=hide, mute=mute,
                                noise_type=noise_type, noise_dimensions=noise_dimensions, normalize=normalize,
                                node_height=400)
        elif type == "TEX_WHITE_NOISE":
            noise_dimensions = attributes["noise_dimensions"]
            return WhiteNoise(tree, location=location, label=label, name=name, hide=hide, mute=mute,
                              noise_dimensions=noise_dimensions, node_height=200)
        elif type == "VALTORGB":
            color_dictionary = attributes["color_ramp"]
            interpolation = attributes["interpolation"]
            color_mode = attributes["color_mode"]
            return ColorRamp(tree, location=location, label=label, name=name, hide=hide, mute=mute,
                             color_dictionary=color_dictionary, interpolation=interpolation, color_mode=color_mode,
                             node_height=400)
        elif type == "VALUE":
            return InputValue(tree, location=location, label=label, name=name, hide=hide, mute=mute, node_height=200)
        elif type == "VECT_MATH":
            operation = attributes["operation"]
            return VectorMathNode(tree, location=location, label=label, name=name, hide=hide, mute=mute,
                                  operation=operation, node_height=200)
        else:
            return None

    def set_parent(self, parent):
        self.node.parent = parent.node


class AttributeNode(ShaderNode):
    def __init__(self, tree, location=(0, 0),
                 attribute_name=None,
                 attribute_type="GEOMETRY",
                 **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeAttribute")
        super().__init__(tree, location, **kwargs)

        self.std_out = self.node.outputs[0]
        self.fac_out = self.node.outputs['Fac']
        self.alpha_aout = self.node.outputs['Alpha']
        self.vector_out = self.node.outputs['Vector']

        if attribute_name:
            self.node.attribute_name = attribute_name
        if attribute_type:
            self.node.attribute_type = attribute_type


class BrightContrast(ShaderNode):
    def __init__(self, tree, location, color=None, bright=0, contrast=0, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeBrightContrast")
        super().__init__(tree, location, **kwargs)

        self.std_out = self.node.outputs["Color"]
        links = tree.links

        if color is not None:
            if isinstance(color, (list, Vector)):
                self.node.inputs["Color"].default_value = color
            else:
                links.new(color, self.node.inputs["Color"])

        if isinstance(bright, (int, float)):
            self.node.inputs["Bright"].default_value = bright
        else:
            links.new(bright, self.node.inputs["Bright"])

        if isinstance(contrast, (int, float)):
            self.node.inputs["Contrast"].default_value = contrast
        else:
            links.new(contrast, self.node.inputs["Contrast"])


class BSDFGlass(ShaderNode):
    def __init__(self, tree, location,
                 distribution="MULTI_GGX",
                 **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeBsdfGlass")
        super().__init__(tree, location, **kwargs)

        self.node.distribution = distribution


class BSDFTransparent(ShaderNode):
    def __init__(self, tree, location, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeBsdfTransparent")
        super().__init__(tree, location, **kwargs)


class Bump(ShaderNode):
    def __init__(self, tree, location, invert=False, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeBump")
        super().__init__(tree, location, **kwargs)

        self.node.invert = invert


class ColorRamp(ShaderNode):
    def __init__(self, tree, location=(0, 0), factor=None, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeValToRGB")
        super().__init__(tree, location, **kwargs)

        if factor is not None:
            if isinstance(factor, (int, float)):
                self.node.inputs[0].default_value = factor
            else:
                self.tree.links.new(factor, self.node.inputs[0])

        if "color_dictionary" in kwargs:
            color_dictionary = kwargs["color_dictionary"]
            # # remove curly braces
            # color_dictionary = color_dictionary[1:-1]
            # dict = {}
            # key_flag = True
            # string = False
            # key_str = ""
            # val_str = ""
            # for l in color_dictionary:
            #     if l == ':' and not string:
            #         key_flag = False
            #     elif l == "'":
            #         if string:
            #             string = False
            #         else:
            #             string = True
            #     elif l == ',' and not string:
            #         key_flag = True
            #         dict[float(key_str)] = parse_vector(val_str)
            #         key_str = ""
            #         val_str = ""
            #     else:
            #         if key_flag:
            #             key_str += l
            #         else:
            #             val_str += l
            # # add last element
            # dict[float(key_str)] = parse_vector(val_str)
            # color_dictionary = dict
            if isinstance(color_dictionary, str):
                color_dictionary = eval(color_dictionary)
                for key, val in color_dictionary.items():
                    color_dictionary[key] = eval(val)
            values = list(color_dictionary.keys())
            colors = list(color_dictionary.values())
        else:
            values = get_from_kwargs(kwargs, "values", [0, 1])
            colors = get_from_kwargs(kwargs, "colors", [[0, 0, 0, 1], [1, 1, 1, 1]])

        for i in range(2, len(values)):
            self.node.color_ramp.elements.new(i)

        i = 0
        for value, color in zip(values, colors):
            self.node.color_ramp.elements[i].position = value
            if len(color) == 3:
                color += [1]  # add default alpha if necessary
            self.node.color_ramp.elements[i].color = color
            i += 1

        interpolation = get_from_kwargs(kwargs, "interpolation", "LINEAR")
        self.node.color_ramp.interpolation = interpolation
        color_mode = get_from_kwargs(kwargs, "color_mode", "RGB")
        self.node.color_ramp.color_mode = color_mode

        self.std_out = self.node.outputs[0]
        self.color_ramp = self.node.color_ramp


class Displacement(ShaderNode):
    def __init__(self, tree, location=(0, 0),
                 height=0,
                 midlevel=0.5,
                 scale=1.,
                 space="OBJECT", **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeDisplacement")
        self.node.space = space
        super().__init__(tree, location, **kwargs)

        self.height = self.node.inputs[0]
        self.midlevel = self.node.inputs[1]
        self.scale = self.node.inputs[2]
        self.std_out = self.node.outputs[0]

        if isinstance(midlevel, (int, float)):
            self.midlevel.default_value = midlevel
        else:
            self.tree.links.new(midlevel, self.midlevel)

        if isinstance(scale, (int, float)):
            self.scale.default_value = scale
        else:
            self.tree.links.new(scale, self.scale)

        if isinstance(height, (int, float)):
            self.height.default_value = height
        else:
            self.tree.links.new(height, self.height)


class GradientTexture(ShaderNode):
    def __init__(self, tree, location=(0, 0), gradient_type='LINEAR', vector=None,
                 std_out='Color', **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeTexGradient")
        super().__init__(tree, location, **kwargs)

        self.node.gradient_type = gradient_type

        if vector is not None:
            if isinstance(vector, (list, Vector)):
                self.node.inputs['Vector'].default_value = vector
            else:
                self.tree.links.new(vector, self.node.inputs['Vector'])

        self.std_out = self.node.outputs[std_out]


class HueSaturationValueNode(ShaderNode):
    def __init__(self, tree, location=(0, 0), hue=0, saturation=1, value=1, fac=1, color=[1, 0, 0, 0], **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeHueSaturation")
        super().__init__(tree, location, **kwargs)

        if isinstance(fac, (int, float)):
            self.node.inputs['Fac'].default_value = fac
        else:
            self.tree.links.new(fac, self.node.inputs['Fac'])

        if isinstance(value, (int, float)):
            self.node.inputs["Value"].default_value = value
        else:
            self.tree.links.new(value, self.node.inputs["Value"])

        if isinstance(saturation, (int, float)):
            self.node.inputs["Saturation"].default_value = saturation
        else:
            self.tree.links.new(saturation, self.node.inputs["Saturation"])

        if isinstance(hue, (int, float)):
            self.node.inputs["Hue"].default_value = hue
        else:
            self.tree.links.new(hue, self.node.inputs["Hue"])

        if isinstance(color, (list, Vector)):
            self.node.inputs["Color"].default_value = color
        else:
            self.tree.links.new(color, self.node.inputs["Color"])

        self.std_out = self.node.outputs[0]


class ImageTexture(ShaderNode):
    def __init__(self, tree, location=(0, 0), image=None, vector=None,
                 std_out='Color', **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeTexImage")
        super().__init__(tree, location, **kwargs)

        if image:
            self.node.image = image

        if vector is not None:
            if isinstance(vector, (list, Vector)):
                self.node.inputs['Vector'].default_value = vector
            else:
                self.tree.links.new(vector, self.node.inputs['Vector'])

        self.std_out = self.node.outputs[std_out]
        self.alpha = self.node.outputs["Alpha"]


class InputValue(ShaderNode):
    def __init__(self, tree, location=(0, 0), value=0, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeValue")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs['Value']
        self.node.outputs['Value'].default_value = value


class MathNode(ShaderNode):
    def __init__(self, tree, location=(0, 0), operation='ADD', input0=None, input1=None, input2=None, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeMath")
        super().__init__(tree, location, **kwargs)

        self.std_out = self.node.outputs["Value"]
        self.node.operation = operation

        if input0:
            if isinstance(input0, (float, int)):
                self.node.inputs[0].default_value = input0
            else:
                self.tree.links.new(input0, self.node.inputs[0])
        if input1:
            if isinstance(input1, (float, int)):
                self.node.inputs[1].default_value = input1
            else:
                self.tree.links.new(input1, self.node.inputs[1])
        if input2:
            if isinstance(input2, (float, int)):
                self.node.inputs[2].default_value = input2
            else:
                self.tree.links.new(input2, self.node.inputs[2])


class VectorMathNode(ShaderNode):
    def __init__(self, tree, location=(0, 0), operation='ADD', input0=None, input1=None, input2=None, scale=None, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeVectorMath")
        super().__init__(tree, location, **kwargs)

        self.std_out = self.node.outputs["Vector"]
        self.node.operation = operation

        if input0:
            if isinstance(input0, (float, int)):
                self.node.inputs[0].default_value = input0
            else:
                self.tree.links.new(input0, self.node.inputs[0])
        if input1:
            if isinstance(input1, (float, int)):
                self.node.inputs[1].default_value = input1
            else:
                self.tree.links.new(input1, self.node.inputs[1])
        if input2:
            if isinstance(input2, (float, int)):
                self.node.inputs[2].default_value = input2
            else:
                self.tree.links.new(input2, self.node.inputs[2])
        if scale is not None:
            if isinstance(scale, (float, int)):
                self.node.inputs["Scale"].default_value = scale
            else:
                self.tree.links.new(scale, self.node.inputs["Scale"])


class Mapping(ShaderNode):
    def __init__(self, tree, location=(0, 0), vector_type='POINT', vector=None,
                 loc=None, rotation=None, scale=None, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeMapping")
        super().__init__(tree, location, **kwargs)

        self.node.vector_type = vector_type

        if vector is not None:
            if isinstance(vector, (list, Vector)):
                self.node.inputs['Vector'].default_value = vector
            else:
                self.tree.links.new(vector, self.node.inputs['Vector'])

        if loc is not None:
            if isinstance(loc, (list, Vector)):
                self.node.inputs['Location'].default_value = loc
            else:
                self.tree.links.new(loc, self.node.inputs['Location'])

        if scale is not None:
            if isinstance(scale, int):
                scale = [scale] * 3
            if isinstance(scale, (list, Vector)):
                self.node.inputs['Scale'].default_value = scale
            else:
                self.tree.links.new(scale, self.node.inputs['Scale'])

        if rotation:
            if isinstance(rotation, (list, Vector)):
                self.node.inputs['Rotation'].default_value = rotation
            else:
                self.tree.links.new(rotation, self.node.inputs['Rotation'])

        self.std_out = self.node.outputs["Vector"]


class NoiseTexture(ShaderNode):
    def __init__(self, tree, location=(0, 0), noise_dimensions='3D', noise_type='FBM', normalize=True,
                 color_dictionary={}, std_out="Color",
                 scale=5, detail=2, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeTexNoise")

        super().__init__(tree, location, **kwargs)

        self.std_out = self.node.outputs[std_out]
        self.node.noise_dimensions = noise_dimensions
        self.node.noise_type = noise_type
        self.node.normalize = normalize
        for key, value in color_dictionary.items():
            self.node.color_ramp.elements[key].color = parse_vector(value)

        if isinstance(scale, (int, float)):
            self.node.inputs['Scale'].default_value = scale
        else:
            self.tree.links.new(scale, self.node.inputs['Scale'])

        if isinstance(detail, (int, float)):
            self.node.inputs['Detail'].default_value = detail


class MixNode(ShaderNode):
    def __init__(self, tree, location, data_type='FLOAT', factor=0, caseA=0, caseB=0, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeMix")
        self.node.data_type = data_type
        self.std_out = self.node.outputs[0]
        super().__init__(tree, location, **kwargs)

        if isinstance(factor, (int, float)):
            self.node.inputs['Factor'].default_value = factor
        else:
            self.tree.links.new(factor, self.node.inputs['Factor'])

        if isinstance(caseA, (int, float)):
            self.node.inputs['A'].default_value = caseA
        else:
            self.tree.links.new(caseA, self.node.inputs['A'])

        if isinstance(caseB, (int, float)):
            self.node.inputs['B'].default_value = caseB
        else:
            self.tree.links.new(caseB, self.node.inputs['B'])


class MixShader(ShaderNode):
    def __init__(self, tree, location, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeMixShader")
        super().__init__(tree, location, **kwargs)


class MixRGB(ShaderNode):
    def __init__(self, tree, location=(0, 0), factor=0, color1=[0, 0, 0, 1], color2=[1, 1, 1, 1], **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeMixRGB")
        super().__init__(tree, location, **kwargs)

        self.factor = self.node.inputs[0]
        self.color1 = self.node.inputs[1]
        self.color2 = self.node.inputs[2]
        self.std_out = self.node.outputs[0]

        if isinstance(factor, (int, float)):
            self.node.inputs[0].default_value = factor
        else:
            self.tree.links.new(factor, self.node.inputs[0])

        if isinstance(color1, (Vector, list)):
            self.node.inputs[1].default_value = color1
        else:
            self.tree.links.new(color1, self.node.inputs[1])

        if isinstance(color2, (Vector, list)):
            self.node.inputs[2].default_value = color2
        else:
            self.tree.links.new(color2, self.node.inputs[2])


class OutputMaterial(ShaderNode):
    def __init__(self, tree, location=(0, 0), **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeOutputMaterial")
        super().__init__(tree, location=location, **kwargs)


class PrincipledBSDF(ShaderNode):
    def __init__(self, tree, location=(0, 0), base_color=None,**kwargs):
        self.node = tree.nodes.new(type="ShaderNodeBsdfPrincipled")
        super().__init__(tree, location=location, **kwargs)

        if base_color:
            tree.links.new(base_color, self.node.inputs["Base Color"])


class PrincipledVolume(ShaderNode):
    def __init__(self, tree, location=(0, 0), **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeVolumePrincipled")
        super().__init__(tree, location=location, **kwargs)


class RGB(ShaderNode):
    def __init__(self, tree, location=(0, 0), color=[0, 0, 0, 1], **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeRGB")
        super().__init__(tree, location=location, **kwargs)

        self.node.outputs["Color"].default_value = color
        self.std_out = self.node.outputs["Color"]


class SeparateXYZ(ShaderNode):
    def __init__(self, tree, location=(0, 0), vector=None, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeSeparateXYZ")
        super().__init__(tree, location=location, **kwargs)

        if vector:
            tree.links.new(vector, self.node.inputs["Vector"])

        self.std_out_x = self.node.outputs['X']
        self.std_out_y = self.node.outputs['Y']
        self.std_out_z = self.node.outputs['Z']


class CombineXYZ(ShaderNode):
    def __init__(self, tree, location=(0, 0), x=0, y=0, z=0, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeCombineXYZ")
        super().__init__(tree, location=location, **kwargs)

        if isinstance(x, (int, float)):
            self.node.inputs[0].default_value = x
        else:
            tree.links.new(x, self.node.inputs["X"])

        if isinstance(y, (int, float)):
            self.node.inputs[1].default_value = y
        else:
            tree.links.new(y, self.node.inputs["Y"])

        if isinstance(z, (int, float)):
            self.node.inputs[2].default_value = z
        else:
            tree.links.new(z, self.node.inputs["Z"])

        self.std_out = self.node.outputs['Vector']


class TextureCoordinate(ShaderNode):
    def __init__(self, tree, location=(0, 0), std_out="UV", **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeTexCoord")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs[std_out]


class TextureImage(ShaderNode):
    def __init__(self, tree, location=(0, 0), std_out="Color",
                 extension="EXTEND", image_name=None, image_source="FILE",
                 image_colorspace_settings_name="sRGB", interpolation="Linear",
                 projection="FLAT", vector=None, **kwargs):

        self.node = tree.nodes.new(type="ShaderNodeTexImage")
        self.node.extension = extension
        if image_name is not None:
            self.node.image = bpy.data.images.load(os.path.join(IMG_DIR, image_name))
            self.node.image.source = image_source
            self.node.image.colorspace_settings.name = image_colorspace_settings_name
        self.node.interpolation = interpolation
        self.node.projection = projection

        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs[std_out]

        if vector is not None:
            if isinstance(vector, (float, Vector)):
                self.node.inputs["Vector"].default_value = vector
            else:
                tree.links.new(vector, self.node.inputs["Vector"])


class WhiteNoise(ShaderNode):
    def __init__(self, tree, location=(0, 0), noise_dimensions="3D", std_out="Value", **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeTexWhiteNoise")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs[std_out]
        self.node.noise_dimensions = noise_dimensions


class EmissionShader(ShaderNode):
    """Wraps ShaderNodeEmission. std_out = outputs["Emission"]."""

    def __init__(self, tree, location=(0, 0), color=None, strength=1.0, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeEmission")
        super().__init__(tree, location=location, **kwargs)

        if color is not None:
            if isinstance(color, (list, Vector)):
                self.node.inputs["Color"].default_value = color
            else:
                self.tree.links.new(color, self.node.inputs["Color"])

        if isinstance(strength, (int, float)):
            self.node.inputs["Strength"].default_value = float(strength)
        else:
            self.tree.links.new(strength, self.node.inputs["Strength"])

        self.std_out = self.node.outputs["Emission"]


class ShaderFrame(ShaderNode):
    """
    Wraps NodeFrame for shader node trees.
    Pass color=(r,g,b) to enable a custom frame colour.
    Use add(*nodes) to make nodes children of this frame; each argument
    may be either a ShaderNode instance (has .node) or a raw bpy node.
    """

    def __init__(self, tree, location=(0, 0), label="Frame", color=None, **kwargs):
        self.node = tree.nodes.new(type="NodeFrame")
        kwargs.setdefault('hide', False)  # frames must not be hidden by default
        super().__init__(tree, location=location, **kwargs)
        self.node.label = label
        if color is not None:
            self.node.use_custom_color = True
            self.node.color = color

    def add(self, *nodes):
        """Parent one or more nodes/ShaderNode instances to this frame."""
        for n in nodes:
            target = n.node if hasattr(n, 'node') else n
            target.parent = self.node


class ShaderRepeatZone(ShaderNode):
    """
    Wraps GeometryNodeRepeatInput/Output in a shader node tree.
    Works identically to RepeatZone in geometry_nodes/nodes.py.

    Usage:
        rz = ShaderRepeatZone(tree, location=(-2, 0), iterations=8, node_width=10)
        rz.add_socket('FLOAT', 'ax')        # add a state variable
        rz.repeat_input.inputs['ax'].default_value = 0.0   # initial value
        # build body between rz.repeat_input and rz.repeat_output
        tree.links.new(out, rz.repeat_output.inputs['ax'])  # wire result
        rz.repeat_output.outputs['ax']      # final value after all iterations
    """

    def __init__(self, tree, location=(0, 0), node_width=10, iterations=10, **kwargs):
        self.repeat_output = tree.nodes.new("GeometryNodeRepeatOutput")
        self.repeat_input = tree.nodes.new("GeometryNodeRepeatInput")
        self.repeat_input.location = (location[0] * 200, location[1] * 100)
        self.repeat_output.location = (location[0] * 200 + node_width * 200, location[1] * 100)
        self.repeat_input.pair_with_output(self.repeat_output)
        self.node = self.repeat_input
        self.iteration = self.repeat_input.outputs["Iteration"]
        kwargs.setdefault('hide', False)
        super().__init__(tree, location=location, **kwargs)

        if isinstance(iterations, int):
            self.repeat_input.inputs["Iterations"].default_value = iterations
        else:
            tree.links.new(iterations, self.repeat_input.inputs["Iterations"])

    def add_socket(self, socket_type="FLOAT", name="socket"):
        """Add a state variable socket (appears on both RepeatInput outputs and RepeatOutput inputs/outputs).

        :param socket_type: 'FLOAT', 'INT', 'BOOLEAN', 'VECTOR', 'RGBA', etc.
        :param name: socket name
        """
        self.repeat_output.repeat_items.new(socket_type, name)


class ShaderClosureZone(ShaderNode):
    """
    Wraps NodeClosureInput/Output in a shader node tree.
    The closure captures input values (available inside the body as closure_input.outputs)
    and exposes output values (plugged into closure_output.inputs, read via EvaluateClosure).

    Usage:
        cz = ShaderClosureZone(tree, location=(-4, 0), node_width=15)
        cz.add_input('FLOAT', 'tp_x')    # captured value → cz.closure_input.outputs['tp_x']
        cz.add_output('FLOAT', 'result') # body result   → cz.closure_output.inputs['result']
        # build body between closure_input and closure_output
        tree.links.new(body_result, cz.closure_output.inputs['result'])
        # evaluate elsewhere:
        ec = EvaluateClosure(tree, closure=cz.std_out, ...)
    """

    def __init__(self, tree, location=(0, 0), node_width=15, **kwargs):
        self.closure_output = tree.nodes.new("NodeClosureOutput")
        self.closure_input = tree.nodes.new("NodeClosureInput")
        self.closure_input.location = (location[0] * 200, location[1] * 100)
        self.closure_output.location = (location[0] * 200 + node_width * 200, location[1] * 100)
        self.closure_input.pair_with_output(self.closure_output)
        self.node = self.closure_input
        self.std_out = self.closure_output.outputs["Closure"]
        kwargs.setdefault('hide', False)
        super().__init__(tree, location=location, **kwargs)

    def add_input(self, socket_type="FLOAT", name="socket"):
        """Capture a value from outside — appears as closure_input.outputs[name] inside the body."""
        self.closure_output.input_items.new(socket_type, name)

    def add_output(self, socket_type="FLOAT", name="socket"):
        """Expose a computed value — plug into closure_output.inputs[name], read via EvaluateClosure."""
        self.closure_output.output_items.new(socket_type, name)


class EvaluateClosure(ShaderNode):
    """
    Wraps NodeEvaluateClosure in a shader node tree.
    Takes a Closure socket, passes input values in, and returns output values.

    Usage:
        ec = EvaluateClosure(tree, closure=cz.std_out, location=(2, 0))
        ec.add_input('FLOAT', 'tp_x')    # maps to cz.closure_input.outputs['tp_x']
        ec.add_output('FLOAT', 'result') # maps to cz.closure_output.inputs['result']
        tree.links.new(g_select.outputs['tp_x'], ec.node.inputs['tp_x'])
        ec.node.outputs['result']        # read the result
    """

    def __init__(self, tree, location=(0, 0), closure=None, **kwargs):
        self.node = tree.nodes.new("NodeEvaluateClosure")
        kwargs.setdefault('hide', False)
        super().__init__(tree, location=location, **kwargs)
        if closure is not None:
            tree.links.new(closure, self.node.inputs["Closure"])
        self.std_out = self.node.outputs

    def add_input(self, socket_type="FLOAT", name="socket"):
        """Add a data input that is passed into the closure body at evaluation time."""
        self.node.input_items.new(socket_type, name)

    def add_output(self, socket_type="FLOAT", name="socket"):
        """Add a data output that receives the result computed inside the closure body."""
        self.node.output_items.new(socket_type, name)


#------------------------
# Custom Node Groups
#------------------------

def _make_new_socket(tree, name='mySocket', io='INPUT', type='NodeSocketFloat'):
    """

    :param tree:
    :param name:
    :param io:
    :param type: select one of the following sockets
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


class ShaderNodeGroup(ShaderNode):
    def __init__(self, tree, **kwargs):
        self.node = None
        self.group_outputs = None
        self.group_inputs = None
        self.group_tree = None
        inputs = get_from_kwargs(kwargs, "inputs", {"Position": "VECTOR", "Index": "INT"})
        outputs = get_from_kwargs(kwargs, "outputs", {"result": "INT"})
        self.create_node_group(tree, inputs, outputs, **kwargs)

        # filled by subclass
        self.fill_group_with_node(self.group_tree, **kwargs)

        auto_layout = get_from_kwargs(kwargs, "auto_layout", True)
        if auto_layout:
            layout(self.group_tree)
        super().__init__(tree, location=get_from_kwargs(kwargs, "location", (0, 0)), **kwargs)

    def create_node_group(self, tree, inputs, outputs, **kwargs):
        # new group and inputs and outputs
        nodes = tree.nodes
        name = get_from_kwargs(kwargs, "name", "DefaultShaderNodeGroup")
        group = nodes.new(type='ShaderNodeGroup')
        node_tree = bpy.data.node_groups.new(name, type='ShaderNodeTree')
        group.node_tree = node_tree
        self.group_tree = node_tree
        nodes = node_tree.nodes
        group.name = name
        self.group_inputs = nodes.new('NodeGroupInput')
        self.group_inputs.location = (-200, 0)
        self.group_outputs = nodes.new('NodeGroupOutput')
        self.group_outputs.location = (200, 0)

        for name, tpe in inputs.items():
            _make_new_socket(node_tree, name=name, io="INPUT", type=SOCKET_TYPES[tpe])
        for name, tpe in outputs.items():
            _make_new_socket(node_tree, name=name, io="OUTPUT", type=SOCKET_TYPES[tpe])

        self.node = group
        super().__init__(tree, **kwargs)

    def fill_group_with_node(self, group_tree, **kwargs):
        """ filled by sub classes """
        pass


class OnRightNode(ShaderNodeGroup):
    def __init__(self, tree, **kwargs):
        super().__init__(tree, inputs={"A": "VECTOR", "B": "VECTOR", "Position": "VECTOR"}, outputs={"Result": "INT"},
                         name="OnRightNode", **kwargs)

        self.inputs = self.node.inputs
        self.outputs = self.node.outputs
        self.std_out = self.node.outputs["Result"]

    def fill_group_with_node(self, group_tree, **kwargs):
        on_right = make_function(group_tree, name="Function",
                                 functions={
                                    "result":"pos,a,sub,pos,b,sub,cross,e_z,dot"
                                 }, inputs=["a","b","pos"], outputs=["result"],
                                 scalars=["result"], vectors=["a","b","pos"],node_group_type="Shader")
        group_tree.links.new(self.group_inputs.outputs["A"], on_right.inputs["a"])
        group_tree.links.new(self.group_inputs.outputs["B"], on_right.inputs["b"])
        group_tree.links.new(self.group_inputs.outputs["Position"], on_right.inputs["pos"])

        group_tree.links.new( on_right.outputs["result"],self.group_outputs.inputs["Result"])


class IfNode(ShaderNodeGroup):
    """
    Ternary selector: Result = Yes if Condition else No.

    Condition is interpreted as a 0/1 scalar (any non-zero treated as true via
    the blend Condition*Yes + (1-Condition)*No).  The data_type kwarg picks the
    socket type used for Yes/No/Result; sockets and internal wiring are
    generated accordingly.
    """
    _ALLOWED_DATA_TYPES = ("INT", "FLOAT", "VECTOR")

    def __init__(self, tree, data_type='FLOAT', **kwargs):
        if data_type not in self._ALLOWED_DATA_TYPES:
            raise ValueError(
                f"IfNode data_type must be one of {self._ALLOWED_DATA_TYPES}, got {data_type!r}"
            )
        self._data_type = data_type
        super().__init__(
            tree,
            inputs={"Condition": "FLOAT", "Yes": data_type, "No": data_type},
            outputs={"Result": data_type},
            name="IfNode",
            **kwargs,
        )
        self.inputs = self.node.inputs
        self.outputs = self.node.outputs
        self.std_out = self.node.outputs["Result"]

    def fill_group_with_node(self, group_tree, **kwargs):
        cond = self.group_inputs.outputs["Condition"]
        yes = self.group_inputs.outputs["Yes"]
        no = self.group_inputs.outputs["No"]
        result_in = self.group_outputs.inputs["Result"]

        if self._data_type == "VECTOR":
            # result = yes*cond + no*(1-cond)  (component-wise via SCALE + ADD)
            scale_yes = VectorMathNode(group_tree, operation='SCALE', input0=yes, scale=cond)
            one_minus_cond = MathNode(group_tree, operation='SUBTRACT', input0=1.0, input1=cond)
            scale_no = VectorMathNode(group_tree, operation='SCALE', input0=no, scale=one_minus_cond.std_out)
            add_v = VectorMathNode(group_tree, operation='ADD', input0=scale_yes.std_out, input1=scale_no.std_out)
            group_tree.links.new(add_v.std_out, result_in)
        else:
            # result = cond*yes + (1-cond)*no
            mul_yes = MathNode(group_tree, operation='MULTIPLY', input0=cond, input1=yes)
            one_minus_cond = MathNode(group_tree, operation='SUBTRACT', input0=1.0, input1=cond)
            mul_no = MathNode(group_tree, operation='MULTIPLY', input0=one_minus_cond.std_out, input1=no)
            add = MathNode(group_tree, operation='ADD', input0=mul_yes.std_out, input1=mul_no.std_out)
            group_tree.links.new(add.std_out, result_in)
