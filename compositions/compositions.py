import bpy

from utils.constants import DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs


###############
# composition #
###############

def create_composition(denoising=None):
    """
    create after render image processing
    :param denoising: toggle denoiser
    :return:
    """

    bpy.context.scene.use_nodes = True
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links

    composite = nodes["Composite"]
    composite.use_alpha=False
    layers = nodes["Render Layers"]
    if denoising:
        denoise = nodes.new(type="CompositorNodeDenoise")

        links.new(layers.outputs["Image"], denoise.inputs["Image"])
        links.new(denoise.outputs["Image"], composite.inputs["Image"])

    alpha_convert = nodes.new(type="CompositorNodePremulKey")
    links.new(alpha_convert.outputs["Image"], composite.inputs["Alpha"])

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

    glare = nodes.new(type='CompositorNodeGlare')
    glare.glare_type='SIMPLE_STAR'
    glare.quality='HIGH'
    glare.threshold = 0.01

    links.new(layers.outputs["Image"],glare.inputs["Image"])
    links.new(glare.outputs["Image"],composite.inputs["Image"])

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

    glare = nodes.new(type='CompositorNodeGlare')
    glare.glare_type = type
    glare.quality='HIGH'
    glare.size=size
    glare.threshold =threshold

    links.new(layers.outputs["Image"],glare.inputs["Image"])
    links.new(glare.outputs["Image"],composite.inputs["Image"])

    return glare

