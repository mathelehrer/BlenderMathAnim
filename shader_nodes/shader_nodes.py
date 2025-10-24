import os

import bpy
import numpy as np
from mathutils import Vector

from geometry_nodes.nodes import Frame
from utils.constants import IMG_DIR
from utils.kwargs import get_from_kwargs
from utils.string_utils import parse_vector

pi = np.pi

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


class ShaderNode:
    def __init__(self, tree, location=(0, 0), width=200, height=100, **kwargs):
        self.tree = tree
        self.location = location
        self.l, self.m = location
        self.node.location = (self.l * width, self.m * height)
        self.links = tree.links
        self.node.hide = True
        self.inputs = self.node.inputs
        self.outputs= self.node.outputs

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
    def from_attributes(cls,tree,attributes):
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

        if type=="ATTRIBUTE":
            attribute_name= attributes["attribute_name"]
            return AttributeNode(tree,location=location,label=label,name=name,attribute_name=attribute_name,hide=hide,mute=mute,node_height=200)
        elif type=="BRIGHTCONTRAST":
            return BrightContrast(tree,location=location,label=label,name=name,hide=hide,mute=mute,node_height=200)
        elif type =="BSDF_PRINCIPLED":
            return PrincipledBSDF(tree,location=location,label=label,
                                  name=name,hide=hide,mute=mute,
                                  node_height=600)
        elif type=="BSDF_TRANSPARENT":
            return BSDFTransparent(tree,location=location,label=label,name=name,hide=hide,mute=mute,
                                   node_height=200)
        elif type == "COMBXYZ":
            return CombineXYZ(tree, location=location, label=label, name=name, hide=hide, mute=mute, node_height=200)
        elif type=="DISPLACEMENT":
            space=attributes["space"]
            return Displacement(tree,location=location,label=label,name=name,hide=hide,mute=mute,
                                 space=space,node_height=400)
        elif type=="FRAME":
            return Frame(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,node_width=200)
        elif type=="MAPPING":
            vector_type=attributes["vector_type"]
            return Mapping(tree,location,vector_type=vector_type,label=label,hide=hide,mute=mute,node_height=200,node_width=200)
        elif type=="MATH":
            operation = attributes["operation"]
            return MathNode(tree,location=location,label=label,name=name,hide=hide,mute=mute,
                            operation=operation,node_height=200)
        elif type=="MIX":
            data_type=attributes["data_type"]
            return MixNode(tree,location=location,label=label,name=name,hide=hide,mute=mute,
                           data_type=data_type,node_height=200)
        elif type=="MIX_SHADER":
            return MixShader(tree,location=location,label=label,name=name,hide=hide,mute=mute,node_height=200)
        elif type=="OUTPUT_MATERIAL":
            return OutputMaterial(tree,location=location,label=label,name=name,hide=hide,mute=mute,node_height=200)
        elif type=="SEPXYZ":
            return SeparateXYZ(tree,location=location,label=label,name=name,hide=hide,mute=mute,node_height=200)
        elif type=="TEX_COORD":
            return TextureCoordinate(tree,location=location,label=label,name=name,hide=hide,mute=mute,node_height=200)
        elif type=="TEX_IMAGE":
            extension = attributes["extension"]
            image_name= attributes["image.name"]
            image_source = attributes["image.source"]
            image_colorspace_settings_name=attributes["image.colorspace_settings.name"]
            interpolation = attributes["interpolation"]
            projection = attributes["projection"]
            return TextureImage(tree,location=location,label=label,name=name,hide=hide,mute=mute,
                                extension=extension,image_name=image_name,image_source=image_source,
                                image_colorspace_settings_name=image_colorspace_settings_name,
                                interpolation=interpolation,projection=projection)
        elif type=="TEX_NOISE":
            noise_dimensions = attributes["noise_dimensions"]
            noise_type = attributes["noise_type"]
            bool_str = attributes["normalize"]
            if bool_str=="True":
                normalize=True
            else:
                normalize=False
            return NoiseTexture(tree,location=location,label=label,name=name,hide=hide,mute=mute,
                                noise_type=noise_type,noise_dimensions=noise_dimensions,normalize=normalize,node_height=400)
        elif type=="VALTORGB":
            color_dictionary = attributes["color_ramp"]
            interpolation=attributes["interpolation"]
            color_mode=attributes["color_mode"]
            return ColorRamp(tree,location=location,label=label,name=name,hide=hide,mute=mute,
                             color_dictionary=color_dictionary,interpolation=interpolation,color_mode=color_mode,
                             node_height=400)
        elif type=="VALUE":
            return InputValue(tree,location=location,label=label,name=name,hide=hide,mute=mute,node_height=200)
        elif type == "VECT_MATH":
            operation = attributes["operation"]
            return VectorMathNode(tree, location=location, label=label, name=name, hide=hide, mute=mute,
                                  operation=operation, node_height=200)
        else:
            return None

    def set_parent(self,parent):
        self.node.parent=parent.node


class AttributeNode(ShaderNode):
    def __init__(self, tree, location=(0, 0), type='GEOMETRY', attribute_name=None, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeAttribute")
        super().__init__(tree, location, **kwargs)

        self.std_out = self.node.outputs[0]
        self.fac_out = self.node.outputs['Fac']
        self.alpha_aout = self.node.outputs['Alpha']
        self.vector_out = self.node.outputs['Vector']

        if attribute_name:
            self.node.attribute_name = attribute_name
        self.node.attribute_type = type

class BrightContrast(ShaderNode):
    def __init__(self, tree, location,color=None,bright=0,contrast = 0, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeBrightContrast")
        super().__init__(tree, location, **kwargs)

        self.std_out = self.node.outputs["Color"]
        links = tree.links

        if color is not None:
            if isinstance(color,(list,Vector)):
                self.node.inputs["Color"].default_value = color
            else:
                links.new(color,self.node.inputs["Color"])

        if isinstance(bright,(int,float)):
            self.node.inputs["Bright"].default_value=bright
        else:
            links.new(bright,self.node.inputs["Bright"])

        if isinstance(contrast,(int,float)):
            self.node.inputs["Contrast"].default_value=contrast
        else:
            links.new(contrast,self.node.inputs["Contrast"])



class BSDFTransparent(ShaderNode):
    def __init__(self,tree,location,**kwargs):
        self.node = tree.nodes.new(type="ShaderNodeBsdfTransparent")
        super().__init__(tree,location,**kwargs)

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
            # remove curly braces
            color_dictionary = color_dictionary[1:-1]
            dict ={}
            key_flag=True
            string=False
            key_str=""
            val_str=""
            for l in color_dictionary:
                if l==':' and not string:
                    key_flag=False
                elif l=="'":
                    if string:
                        string=False
                    else:
                        string=True
                elif l==',' and not string:
                    key_flag=True
                    dict[float(key_str)]=parse_vector(val_str)
                    key_str=""
                    val_str=""
                else:
                    if key_flag:
                        key_str+=l
                    else:
                        val_str+=l
            # add last element
            dict[float(key_str)]=parse_vector(val_str)
            color_dictionary=dict
            values = list(color_dictionary.keys())
            colors = list(color_dictionary.values())
        else:
            values = get_from_kwargs(kwargs,"values",[0,1])
            colors = get_from_kwargs(kwargs,"colors",[[0,0,0,1],[1,1,1,1]])

        for i in range(2,len(values)):
            self.node.color_ramp.elements.new(i)

        i = 0
        for value,color in zip(values,colors):
            self.node.color_ramp.elements[i].position=value
            if len(color)==3:
                color+=[1] # add default alpha if necessary
            self.node.color_ramp.elements[i].color=color
            i+=1

        interpolation=get_from_kwargs(kwargs,"interpolation","LINEAR")
        self.node.color_ramp.interpolation = interpolation
        color_mode=get_from_kwargs(kwargs,"color_mode","RGB")
        self.node.color_ramp.color_mode = color_mode

        self.std_out = self.node.outputs[0]
        self.color_ramp = self.node.color_ramp

class Displacement(ShaderNode):
    def __init__(self, tree, location=(0, 0),
                 height=0,
                 midlevel=0.5,
                 scale= 1.,
                 space="OBJECT", **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeDisplacement")
        self.node.space = space
        super().__init__(tree, location, **kwargs)

        self.height=self.node.inputs[0]
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
            self.node.image=image

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
    def __init__(self, tree, location=(0, 0), operation='ADD', input0=None, input1=None, input2=None, **kwargs):
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
    def __init__(self, tree, location=(0, 0),noise_dimensions='3D',noise_type='FBM',normalize=True,
                 color_dictionary={}, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeTexNoise")

        super().__init__(tree, location, **kwargs)

        self.node.noise_dimensions=noise_dimensions
        self.node.noise_type=noise_type
        self.node.normalize=normalize
        for key,value in color_dictionary.items():
            self.node.color_ramp.elements[key].color=parse_vector(value)

class MixNode(ShaderNode):
    def __init__(self,tree,location,data_type='FLOAT',factor=0,caseA=0,caseB=0,**kwargs):
        self.node = tree.nodes.new(type="ShaderNodeMix")
        self.node.data_type=data_type
        self.std_out = self.node.outputs[0]
        super().__init__(tree,location,**kwargs)

        if isinstance(factor,(int,float)):
            self.node.inputs['Factor'].default_value=factor
        else:
            self.tree.links.new(factor,self.node.inputs['Factor'])

        if isinstance(caseA,(int,float)):
            self.node.inputs['A'].default_value=caseA
        else:
            self.tree.links.new(caseA,self.node.inputs['A'])

        if isinstance(caseB, (int, float)):
            self.node.inputs['B'].default_value = caseB
        else:
            self.tree.links.new(caseB, self.node.inputs['B'])

class MixShader(ShaderNode):
    def __init__(self,tree,location,**kwargs):
        self.node = tree.nodes.new(type="ShaderNodeMixShader")
        super().__init__(tree,location,**kwargs)

class MixRGB(ShaderNode):
    def __init__(self, tree, location=(0, 0), factor=0, color1=[0, 0, 0, 1], color2=[1, 1, 1, 1], **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeMixRGB")
        super().__init__(tree, location, **kwargs)

        self.factor=self.node.inputs[0]
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
    def __init__(self, tree, location=(0, 0),  **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeOutputMaterial")
        super().__init__(tree, location=location, **kwargs)

class PrincipledBSDF(ShaderNode):
    def __init__(self, tree, location=(0, 0),  **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeBsdfPrincipled")
        super().__init__(tree, location=location, **kwargs)

class SeparateXYZ(ShaderNode):
    def __init__(self, tree, location=(0, 0), vector=None, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeSeparateXYZ")
        super().__init__(tree, location=location, **kwargs)

        if vector:
            tree.links.new(vector,self.node.inputs["Vector"])

        self.std_out_x = self.node.outputs['X']
        self.std_out_y = self.node.outputs['Y']
        self.std_out_z = self.node.outputs['Z']

class CombineXYZ(ShaderNode):
    def __init__(self, tree, location=(0, 0), x=0,y=0,z=0, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeCombineXYZ")
        super().__init__(tree, location=location, **kwargs)

        if isinstance(x,(int,float)):
            self.node.inputs[0].default_value = x
        else:
            tree.links.new(x,self.node.inputs["X"])

        if isinstance(y,(int,float)):
            self.node.inputs[1].default_value = y
        else:
            tree.links.new(y,self.node.inputs["Y"])


        if isinstance(z,(int,float)):
            self.node.inputs[2].default_value = z
        else:
            tree.links.new(z,self.node.inputs["Z"])

        self.std_out = self.node.outputs['Vector']


class TextureCoordinate(ShaderNode):
    def __init__(self, tree, location=(0, 0), std_out="UV", **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeTexCoord")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs[std_out]


class TextureImage(ShaderNode):
    def __init__(self, tree, location=(0, 0), std_out="Color",
                 extension="EXTEND",image_name=None,image_source="FILE",
                 image_colorspace_settings_name="sRGB",interpolation="Linear",
                 projection="FLAT",vector=None,**kwargs):

        self.node = tree.nodes.new(type="ShaderNodeTexImage")
        self.node.extension=extension
        if image_name is not None:
            self.node.image=bpy.data.images.load(os.path.join(IMG_DIR,image_name))
            self.node.image.source=image_source
            self.node.image.colorspace_settings.name=image_colorspace_settings_name
        self.node.interpolation=interpolation
        self.node.projection=projection

        super().__init__(tree,location=location,**kwargs)

        self.std_out = self.node.outputs[std_out]

        if vector is not None:
            if isinstance(vector,(float,Vector)):
                self.node.inputs["Vector"].default_value=vector
            else:
                tree.links.new(vector,self.node.inputs["Vector"])


