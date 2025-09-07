import bpy

from appearance.textures import get_color
from utils.constants import DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs


###############
# composition #
###############

def create_alpha_over_composition(color="background"):
    bpy.context.scene.use_nodes = True
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links

    composite = nodes["Composite"]
    layers = nodes["Render Layers"]

    box_mask = nodes.new(type="CompositorNodeBoxMask")
    box_mask.x=0.5
    box_mask.y=0.65
    box_mask.mask_width=0.9
    box_mask.mask_height=0.6

    blur = nodes.new(type="CompositorNodeBlur")
    blur.size_x=1000
    blur.size_y=1000
    links.new(box_mask.outputs['Mask'],blur.inputs['Image'])

    mix = nodes.new(type="CompositorNodeMixRGB")
    mix.inputs[1].default_value=get_color("background")
    mix.inputs[2].default_value=get_color("gray_1")
    links.new(blur.outputs["Image"],mix.inputs[0])

    alpha_over = nodes.new(type="CompositorNodeAlphaOver")
    links.new(mix.outputs["Image"],alpha_over.inputs[1])
    links.new(layers.outputs["Image"],alpha_over.inputs[2])
    links.new(alpha_over.outputs["Image"],composite.inputs[0])


    viewer = nodes.new(type="CompositorNodeViewer")
    links.new(alpha_over.outputs["Image"], viewer.inputs[0])

def create_star_glow_composition():
    """
    create after render image processing
    :return:
    """

    bpy.context.scene.use_nodes = True
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links

    composite = nodes["Composite"]
    layers = nodes["Render Layers"]
    set_alpha = nodes["Set Alpha"]
    viewer = nodes["Viewer"]
    nodes.remove(viewer)

    glare = nodes.new(type='CompositorNodeGlare')
    glare.glare_type='SIMPLE_STAR'
    glare.quality='HIGH'
    glare.threshold = 0.01

    links.new(layers.outputs["Image"], glare.inputs["Image"])
    links.new(layers.outputs["Alpha"], set_alpha.inputs["Alpha"])
    set_alpha.mode = "REPLACE_ALPHA"
    links.new(glare.outputs["Image"],set_alpha.inputs["Image"])
    links.new(set_alpha.outputs["Image"], composite.inputs["Image"])
    return glare


def create_glow_composition(threshold=1,type='BLOOM',size=4):
    """
    create after render image processing
    :return:
    """

    bpy.context.scene.use_nodes = True
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links

    composite = nodes["Composite"]
    layers = nodes["Render Layers"]
    set_alpha = nodes["Set Alpha"]
    viewer = nodes["Viewer"]
    nodes.remove(viewer)

    glare = nodes.new(type='CompositorNodeGlare')
    glare.glare_type = type
    glare.quality='HIGH'
    glare.size=size*1000
    glare.threshold =threshold

    links.new(layers.outputs["Image"],glare.inputs["Image"])
    links.new(layers.outputs["Alpha"],set_alpha.inputs["Alpha"])
    set_alpha.mode="REPLACE_ALPHA"
    links.new(glare.outputs["Image"],set_alpha.inputs["Image"])
    links.new(set_alpha.outputs["Image"],composite.inputs["Image"])
    return glare

def create_bloom_and_streak_composition():
    """
    create after render image processing
    :return:
    """

    bpy.context.scene.use_nodes = True
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links

    composite = nodes["Composite"]
    layers = nodes["Render Layers"]
    alpha = nodes.new(type="CompositorNodeSetAlpha")
    glare = nodes.new(type='CompositorNodeGlare')
    glare.glare_type = "BLOOM"
    glare.quality='HIGH'
    glare.size=8
    glare.threshold =1

    glare2 = nodes.new(type='CompositorNodeGlare')
    glare2.glare_type="STREAKS"
    glare2.quality='HIGH'

    mix = nodes.new(type="CompositorNodeMixRGB")
    mix.inputs["Fac"].default_value=0.75
    links.new(glare.outputs["Image"],mix.inputs[1])
    links.new(glare2.outputs["Image"],mix.inputs[2])


    links.new(layers.outputs["Image"],glare.inputs["Image"])
    links.new(layers.outputs["Image"],glare2.inputs["Image"])
    links.new(layers.outputs["Alpha"],alpha.inputs["Image"])
    links.new(mix.outputs["Image"],composite.inputs["Image"])
    set_alpha = nodes.new(type="CompositorNodeSetAlpha")
    set_alpha.mode = "REPLACE"
    links.new(mix.outputs["Image"], set_alpha.inputs["Image"])
    links.new(set_alpha.outputs["Image"], composite.inputs["Alpha"])
    return glare

    return glare