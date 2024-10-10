import numpy as np
from mathutils import Vector

from utils.kwargs import get_from_kwargs

pi = np.pi


class ShaderNode:
    def __init__(self, tree, location=(0, 0), width=200, height=100, **kwargs):
        self.tree = tree
        self.location = location
        self.l, self.m = location
        self.node.location = (self.l * width, self.m * height)
        self.links = tree.links
        self.node.hide = True

        if 'hide' in kwargs:
            hide = kwargs.pop('hide')
            self.node.hide = hide
        if 'name' in kwargs:
            name = kwargs.pop('name')
            self.node.label = name
            self.node.name = name
        if 'label' in kwargs:
            label = kwargs.pop('label')
            self.node.label = label


class TextureCoordinate(ShaderNode):
    def __init__(self, tree, location=(0, 0), std_out="UV", **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeTexCoord")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs[std_out]

class SeparateXYZ(ShaderNode):
    def __init__(self, tree, location=(0, 0), vector=None, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeSeparateXYZ")
        super().__init__(tree, location=location, **kwargs)

        if vector:
            tree.links.new(vector,self.node.inputs["Vector"])

        self.std_out_x = self.node.outputs['X']
        self.std_out_y = self.node.outputs['Y']
        self.std_out_z = self.node.outputs['Z']


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

class MixNode(ShaderNode):
    def __init__(self,tree,location,type='FLOAT',factor=0,caseA=0,caseB=0,**kwargs):
        self.node = tree.nodes.new(type="ShaderNodeMix")
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

class Mapping(ShaderNode):
    def __init__(self, tree, location=(0, 0), type='POINT', vector=None,
                 loc=None, rotation=None, scale=None, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeMapping")
        super().__init__(tree, location, **kwargs)

        self.node.vector_type = type

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


class ColorRamp(ShaderNode):
    def __init__(self, tree, location=(0, 0), factor=None, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeValToRGB")
        super().__init__(tree, location, **kwargs)

        if isinstance(factor, (int, float)):
            self.node.inputs[0].default_value = factor
        else:
            self.tree.links.new(factor, self.node.inputs[0])

        values = get_from_kwargs(kwargs,"values",[0,1])
        colors = get_from_kwargs(kwargs,"colors",[[0,0,0,1],[1,1,1,1]])

        if len(values)>2:
            self.node.color_ramp.elements.new(len(values)-2)

        i = 0
        for value,color in zip(values,colors):
            self.node.color_ramp.elements[i].position=value
            self.node.color_ramp.elements[i].color=color
            i+=1

        self.std_out = self.node.outputs[0]
        self.color_ramp = self.node.color_ramp


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


class InputValue(ShaderNode):
    def __init__(self, tree, location=(0, 0), value=0, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeValue")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs['Value']
        self.node.outputs['Value'].default_value = value

class Displacement(ShaderNode):
    def __init__(self, tree, location=(0, 0),
                 height=0,
                 midlevel=0.5,
                 scale= 1., **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeDisplacement")
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
