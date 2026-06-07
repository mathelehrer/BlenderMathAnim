import bpy

from appearance.textures import get_color
from interface.interface_constants import blender_version
from utils.constants import DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs
from utils.utils import de_capitalize


###############
# composition #
###############



def create_alpha_over_composition(color="background"):
    if blender_version() >= (5, 1):
        # scene.node_tree / scene.use_nodes were removed in Blender 5.1,
        # the compositor is now a standalone node group datablock
        bpy.ops.node.new_compositing_node_group(name="MyComposition")
        bpy.context.scene.compositing_node_group = bpy.data.node_groups["MyComposition"]
        nodes = bpy.context.scene.compositing_node_group.nodes
        links = bpy.context.scene.compositing_node_group.links
        composite = nodes["Group Output"]
    else:
        bpy.context.scene.use_nodes = True
        nodes = bpy.context.scene.node_tree.nodes
        links = bpy.context.scene.node_tree.links
        composite = nodes["Composite"]

    layers = nodes["Render Layers"]

    box_mask = nodes.new(type="CompositorNodeBoxMask")
    if blender_version() >= (5, 1):
        # x/y/mask_width/mask_height became the Position and Size input sockets
        box_mask.inputs["Position"].default_value = (0.5, 0.65)
        box_mask.inputs["Size"].default_value = (0.9, 0.6)
    else:
        box_mask.x=0.5
        box_mask.y=0.65
        box_mask.mask_width=0.9
        box_mask.mask_height=0.6

    blur = nodes.new(type="CompositorNodeBlur")
    if blender_version() >= (5,1):
        blur.inputs["Size"].default_value = (1000,1000)
    else:
        blur.size_x=1000
        blur.size_y=1000
    links.new(box_mask.outputs['Mask'],blur.inputs['Image'])

    if blender_version() >= (5, 1):
        # CompositorNodeMixRGB was replaced by the generic ShaderNodeMix in Blender 5
        mix = nodes.new(type="ShaderNodeMix")
        mix.data_type = 'RGBA'
        mix.inputs[6].default_value=get_color("background")   # A (color)
        mix.inputs[7].default_value=get_color("gray_1")       # B (color)
        links.new(blur.outputs["Image"],mix.inputs[0])        # Factor
        mix_out = mix.outputs[2]                               # Result (color)
    else:
        mix = nodes.new(type="CompositorNodeMixRGB")
        mix.inputs[1].default_value=get_color("background")
        mix.inputs[2].default_value=get_color("gray_1")
        links.new(blur.outputs["Image"],mix.inputs[0])
        mix_out = mix.outputs["Image"]

    alpha_over = nodes.new(type="CompositorNodeAlphaOver")
    links.new(mix_out,alpha_over.inputs[1])
    links.new(alpha_over.outputs["Image"],composite.inputs[0])

    if blender_version() >= (5, 1):
        viewer = nodes["Viewer"]
    else:
        viewer = nodes.new(type="CompositorNodeViewer")
    links.new(alpha_over.outputs["Image"], viewer.inputs[0])

def create_star_glow_composition():
    """
    create after render image processing
    :return:
    """

    if blender_version() >= (5, 1):
        # scene.node_tree / scene.use_nodes were removed in Blender 5.1,
        # the compositor is now a standalone node group datablock
        bpy.ops.node.new_compositing_node_group(name="MyComposition")
        bpy.context.scene.compositing_node_group = bpy.data.node_groups["MyComposition"]
        nodes = bpy.context.scene.compositing_node_group.nodes
        links = bpy.context.scene.compositing_node_group.links
        composite = nodes["Group Output"]
        set_alpha = nodes.new(type="CompositorNodeSetAlpha")
    else:
        bpy.context.scene.use_nodes = True
        nodes = bpy.context.scene.node_tree.nodes
        links = bpy.context.scene.node_tree.links
        composite = nodes["Composite"]
        set_alpha = nodes["Set Alpha"]
        viewer = nodes["Viewer"]
        nodes.remove(viewer)

    layers = nodes["Render Layers"]

    glare = nodes.new(type='CompositorNodeGlare')
    if blender_version() >= (5, 1):
        glare.inputs["Type"].default_value = "Simple Star"
        glare.inputs["Quality"].default_value = "High"
        glare.inputs["Threshold"].default_value = 0.01
        set_alpha.inputs["Type"].default_value = "Replace Alpha"
    else:
        glare.glare_type='SIMPLE_STAR'
        glare.quality='HIGH'
        glare.threshold = 0.01
        set_alpha.mode = "REPLACE_ALPHA"

    links.new(layers.outputs["Image"], glare.inputs["Image"])
    links.new(glare.outputs["Image"],set_alpha.inputs["Image"])
    links.new(set_alpha.outputs["Image"], composite.inputs["Image"])
    return glare


def create_glow_composition(threshold=1,type='BLOOM',size=4):
    """
    create after render image processing
    :return:
    """

    if blender_version()<(5,0):
        bpy.context.scene.use_nodes = True
        nodes = bpy.context.scene.node_tree.nodes
        links = bpy.context.scene.node_tree.links
    else:
        bpy.ops.node.new_compositing_node_group(name="MyComposition")
        bpy.context.scene.compositing_node_group = bpy.data.node_groups["MyComposition"]
        nodes = bpy.context.scene.compositing_node_group.nodes
        links = bpy.context.scene.compositing_node_group.links


    if blender_version()<(5,0):
        composite = nodes["Composite"]
        set_alpha = nodes["Set Alpha"]
    else:
        set_alpha = nodes.new(type="CompositorNodeSetAlpha")

    layers = nodes["Render Layers"]
    viewer = nodes["Viewer"]
    out = nodes["Group Output"]

    glare = nodes.new(type='CompositorNodeGlare')
    if blender_version()<(5,0):
        glare.glare_type = type
        glare.quality='HIGH'
        glare.size=size
        glare.threshold =threshold
        set_alpha.mode = "REPLACE_ALPHA"
    else:

        glare.inputs["Type"].default_value=de_capitalize(type)
        glare.inputs["Quality"].default_value="High"
        glare.inputs["Size"].default_value=size
        set_alpha.inputs["Type"].default_value="Replace Alpha"

    links.new(layers.outputs["Image"],glare.inputs["Image"])

    links.new(glare.outputs["Image"],set_alpha.inputs["Image"])
    links.new(set_alpha.outputs["Image"],viewer.inputs["Image"])
    links.new(set_alpha.outputs["Image"],out.inputs["Image"])
    return glare

def set_alpha_composition(**kwargs):
    """
    create after render image processing
    :return:
    """

    if blender_version()<(5,0):
        bpy.context.scene.use_nodes = True
        nodes = bpy.context.scene.node_tree.nodes
        links = bpy.context.scene.node_tree.links
    else:
        bpy.ops.node.new_compositing_node_group(name="MyComposition")
        bpy.context.scene.compositing_node_group = bpy.data.node_groups["MyComposition"]
        nodes = bpy.context.scene.compositing_node_group.nodes
        links = bpy.context.scene.compositing_node_group.links


    if blender_version()<(5,0):
        composite = nodes["Composite"]
        set_alpha = nodes["Set Alpha"]
    else:
        set_alpha = nodes.new(type="CompositorNodeSetAlpha")

    layers = nodes["Render Layers"]
    viewer = nodes["Viewer"]
    out = nodes["Group Output"]

    set_alpha.inputs["Type"].default_value="Replace Alpha"

    links.new(layers.outputs["Image"],set_alpha.inputs["Image"])
    links.new(set_alpha.outputs["Image"],viewer.inputs["Image"])
    links.new(set_alpha.outputs["Image"],out.inputs["Image"])

def create_bloom_and_streak_composition():
    """
    create after render image processing
    :return:
    """

    if blender_version() >= (5, 1):
        # scene.node_tree / scene.use_nodes were removed in Blender 5.1,
        # the compositor is now a standalone node group datablock
        bpy.ops.node.new_compositing_node_group(name="MyComposition")
        bpy.context.scene.compositing_node_group = bpy.data.node_groups["MyComposition"]
        nodes = bpy.context.scene.compositing_node_group.nodes
        links = bpy.context.scene.compositing_node_group.links
        composite = nodes["Group Output"]
    else:
        bpy.context.scene.use_nodes = True
        nodes = bpy.context.scene.node_tree.nodes
        links = bpy.context.scene.node_tree.links
        composite = nodes["Composite"]
        set_alpha = nodes["Set Alpha"]
        viewer = nodes["Viewer"]
        nodes.remove(viewer)

    layers = nodes["Render Layers"]

    glare = nodes.new(type='CompositorNodeGlare')
    glare2 = nodes.new(type='CompositorNodeGlare')
    if blender_version() >= (5, 1):
        glare.inputs["Type"].default_value = "Bloom"
        glare.inputs["Quality"].default_value = "High"
        glare.inputs["Size"].default_value = 8000
        glare.inputs["Threshold"].default_value = 1
        glare2.inputs["Type"].default_value = "Streaks"
        glare2.inputs["Quality"].default_value = "High"
    else:
        glare.glare_type = "BLOOM"
        glare.quality='HIGH'
        glare.size=8000
        glare.threshold =1
        glare2.glare_type="STREAKS"
        glare2.quality='HIGH'

    if blender_version() >= (5, 1):
        # CompositorNodeMixRGB was replaced by the generic ShaderNodeMix in Blender 5
        mix = nodes.new(type="ShaderNodeMix")
        mix.data_type = 'RGBA'
        mix.inputs[0].default_value = 0.75          # Factor
        links.new(glare.outputs["Image"], mix.inputs[6])    # A (color)
        links.new(glare2.outputs["Image"], mix.inputs[7])   # B (color)
        mix_out = mix.outputs[2]                     # Result (color)
    else:
        mix = nodes.new(type="CompositorNodeMixRGB")
        mix.inputs["Fac"].default_value=0.75
        links.new(glare.outputs["Image"],mix.inputs[1])
        links.new(glare2.outputs["Image"],mix.inputs[2])
        mix_out = mix.outputs["Image"]

    links.new(layers.outputs["Image"],glare.inputs["Image"])
    links.new(layers.outputs["Image"],glare2.inputs["Image"])

    set_alpha = nodes.new(type="CompositorNodeSetAlpha")
    if blender_version() >= (5, 1):
        set_alpha.inputs["Type"].default_value = "Replace Alpha"
    else:
        set_alpha.mode = "REPLACE_ALPHA"
    links.new(mix_out, set_alpha.inputs["Image"])
    links.new(set_alpha.outputs["Image"], composite.inputs["Image"])
    return glare