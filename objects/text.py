import hashlib
import math
import os
from copy import deepcopy

from geometry_nodes.geometry_nodes_modifier import  GeometryNodesModifier
from geometry_nodes.nodes import InputVector, CollectionInfo, SetMaterial, create_geometry_line, TransformGeometry, \
    create_from_xml
from interface import ibpy
from interface.ibpy import Vector, get_material, get_collection, get_geometry_node_from_modifier
from objects.bobject import BObject
from utils.constants import TEX_LOCAL_SCALE_UP, TEMPLATE_TEXT_FILE, TEMPLATE_TEX_FILE, SVG_DIR, TEX_DIR, \
    TEX_TEXT_TO_REPLACE, DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs


class Text(BObject):
    """
    A new class for a text object based on geometry nodes
    """
    def __init__(self,expression, **kwargs):
        """
        example:
        text = Text("Hallo Welt",color="drawing",outline_color="example",aligned="center",
        emission=0.5,outline_emission=2)
        text.write(begin_time=1,transition_time=1)


        """
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name',"TextObject")
        self.rotation = self.get_from_kwargs('rotation', Vector((math.pi / 2, 0, 0)))

        self.modifier = TextModifier(expression,rotation=self.rotation,**kwargs)
        cube = ibpy.add_cube()
        self.kwargs = kwargs

        #apply colors
        color = get_from_kwargs(kwargs,'color',"text")
        outline_color = get_from_kwargs(kwargs,'outline_color',color)

        mat = get_material(color,**kwargs)
        get_from_kwargs(kwargs,'emission',0) # just remove it from kwargs
        outline_emission = get_from_kwargs(kwargs,'emission_outline',1)
        mat_outline = get_material(outline_color,emission=outline_emission,**kwargs)

        material_node = get_geometry_node_from_modifier(self.modifier,label="FontMaterial")
        outline_material_node = get_geometry_node_from_modifier(self.modifier,label="OutlineMaterial")

        # material_node.inputs['Material'].default_value= mat
        # outline_material_node.inputs['Material'].default_value = mat_outline

        super().__init__(obj=cube, name=self.name, no_material=True, **kwargs)
        super().add_mesh_modifier('NODES', node_modifier=self.modifier)

    def write(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        write_control = get_geometry_node_from_modifier(self.modifier,"WriteControl")
        # ibpy.change_default_value(write_control,from_value=0,to_value=self.modifier.number_of_letters,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

class TextModifier(GeometryNodesModifier):
    def __init__(self,expression,**kwargs):
        self.expression = expression
        self.number_of_letters =0
        super().__init__(get_from_kwargs(kwargs,'name',"GeoText"),
                         group_input=False,group_output=False,automatic_layout=False,**kwargs)

    def create_node(self,tree,**kwargs):

        self.number_of_letters = generate_expression(self.expression, **kwargs)
        create_from_xml(tree,"geo_fonts",**kwargs)

        collection_info = tree.nodes.get("TextData")
        # collection_info.inputs["Separate Children"].default_value = True
        collection_info.inputs["Collection"].default_value = get_collection(self.expression)

        # out = self.group_outputs
        # links = tree.links
        #
        #
        # expr_location = InputVector(tree, name="ExprLocation",
        #                              value=get_from_kwargs(kwargs,'location',Vector()))
        # expr_rotation = InputVector(tree, name="ExprRotation",value=get_from_kwargs(kwargs,'rotation',Vector()))
        # expr_info = CollectionInfo(tree, collection_name=self.expression,
        #                             name="TextData")
        # material = get_material(get_from_kwargs(kwargs,'material',"drawing"))
        # self.materials.append(material)
        # material_node =SetMaterial(tree,material=material)
        # transform_geometry = TransformGeometry(tree,name="ExprTransform",translation=expr_location.std_out,
        #                                        rotation=expr_rotation.std_out)
        # create_geometry_line(tree,[expr_info,transform_geometry,material_node],out=out.inputs[0])


##
# recreate the essentials to convert a latex expression into a collection of curves
# that can be further processed in geometry nodes
# returns the number of letters
##
def generate_expression(expression, **kwargs):
    aligned = get_from_kwargs(kwargs, 'aligned', 'left')
    imported_svg_data = {}  # Build dictionary of imported svgs to use

    if len(expression)>0:
        imported_svg_data={}
        path=get_file_path(expression)
        imported_svg_data = import_svg_data(imported_svg_data,path,kwargs)
        imported_svg_data = align_figures(imported_svg_data, aligned)
        collection = ibpy.make_new_collection(expression,hide_render=True,hide_viewport=True)
        curves = imported_svg_data[list(imported_svg_data.keys())[-1]]
        for curve in curves:
            # make curve dimension 3D to preserve spline property
            curve.data.dimensions = '3D'
            ibpy.link(curve,collection)

        # replace default import collection with a more appropriately named collection
        collection_name = path.split(os.sep)[-1]
        old_collection=ibpy.get_collection(collection_name)
        for obj in old_collection.objects:
            ibpy.un_link(obj, old_collection.name)
        ibpy.remove_collection(old_collection)
        label = collection.name
    return len(curves)

def import_svg_data(imported_svg_data,path,kwargs):
    default_color = get_from_kwargs(kwargs, 'color', 'text')
    text_size = get_from_kwargs(kwargs, 'text_size', 'normal')
    scale = get_from_kwargs(kwargs, 'scale', 1)
    vert_align_centers = get_from_kwargs(kwargs, 'vert_align_centers', 'x_and_y')

    # Import svg and get list of new curves in Blender
    if path not in imported_svg_data.keys():
        imported_svg_data[path] = {'curves': []}

        previous_curves = ibpy.get_all_curves()
        ibpy.import_curve(path)  # here the import takes place
        # all curves are imported into a separate collection with the name path
        new_curves = [x for x in ibpy.get_all_curves() if x not in previous_curves]

        # Arrange new curves relative to tex object's ref_obj
        if text_size == 'normal':
            scale_up = TEX_LOCAL_SCALE_UP
        elif text_size == 'medium':
            scale_up = TEX_LOCAL_SCALE_UP * 1.25
        elif text_size == 'large':
            scale_up = 1.5 * TEX_LOCAL_SCALE_UP
        elif text_size == 'Large':
            scale_up = 2 * TEX_LOCAL_SCALE_UP
        elif text_size == 'Small':
            scale_up = 0.8 * TEX_LOCAL_SCALE_UP
        elif text_size == 'small':
            scale_up = 0.7 * TEX_LOCAL_SCALE_UP
        elif text_size == 'xs':
            scale_up = 0.65 * TEX_LOCAL_SCALE_UP
        elif text_size == 'tiny':
            scale_up = 0.5 * TEX_LOCAL_SCALE_UP
        elif text_size == 'huge':
            scale_up = 3 * TEX_LOCAL_SCALE_UP
        elif text_size == 'Huge':
            scale_up = 5 * TEX_LOCAL_SCALE_UP
        else:  # a number
            scale_up = text_size * TEX_LOCAL_SCALE_UP

        # in the following lines the location of the curve is determined from the geometry and the
        # y-coordinates of the locations are all aligned with the reference 'H'
        for curve in new_curves:
            for spline in curve.data.splines:
                for point in spline.bezier_points:
                    point.handle_left_type = 'FREE'
                    point.handle_right_type = 'FREE'
                # This needs to be in a separate loop because moving points before
                # they're all 'Free' type makes the shape warp.
                # It makes a cool "disappear in the wind" visual, though.
                for point in spline.bezier_points:
                    for i in range(len(point.co)):
                        point.co[i] *= (scale * scale_up)
                        point.handle_left[i] *= (scale * scale_up)
                        point.handle_right[i] *= (scale * scale_up)

            ibpy.set_origin(curve)
            # This part is just meant for tex_objects
            if vert_align_centers:
                loc = curve.location
                new_y = new_curves[0].location[1]  # reference location of the 'H'
                ibpy.set_pivot(curve, [loc[0], new_y, loc[2]])
            ibpy.un_select(curve)

            # find the collection of the curve and unlink object
            # for collection in bpy.context.scene.collection.children:
            #     for o in collection.objects:
            #         if o == curve:
            #             ibpy.un_link(curve, collection.name)
            #             break

            imported_svg_data[path]['curves'] = new_curves
    return imported_svg_data

#### Generating functions
def get_file_path(expression, text_only=False, typeface="default"):
    # Replaces the svg_b_object method
    if text_only:
        template = deepcopy(TEMPLATE_TEXT_FILE)
    else:
        template = deepcopy(TEMPLATE_TEX_FILE)
    if typeface != 'default':
        template = template[:-10]  # chop off the _arial.tex
        template += '_' + typeface + '.tex'
        if not os.path.exists(template):
            raise Warning(r'Can\'t find template tex file for that font.')

    return tex_to_svg_file(expression, template, typeface, text_only)

def tex_to_svg_file(expression, template_tex_file, typeface, text_only):
    path = os.path.join(
        SVG_DIR,
        hashed_tex(expression, typeface)
    ) + ".svg"
    if os.path.exists(path):
        return path

    tex_file = generate_tex_file(expression, template_tex_file, typeface, text_only)
    dvi_file = tex_to_dvi(tex_file)
    return dvi_to_svg(dvi_file)

def generate_tex_file(expression, template_tex_file, typeface, text_only):
    result = os.path.join(
        TEX_DIR,
        # tex_title(expression, typeface)
        hashed_tex(expression, typeface)
    ) + ".tex"

    if not os.path.exists(result):
        print("Writing \"%s\" to %s" % (
            "".join(expression), result
        ))
        with open(template_tex_file, "r") as infile:
            body = infile.read()
            # I add an H to every expression to give a common reference point
            # for all expressions, then hide the H character. This is necessary
            # for consistent alignment of tex curves in blender, because
            # blender's import svg bobject sets the object's origin depending
            # on the expression itself, not according to a typesetting reference
            # frame.
            if text_only:
                expression = 'H ' + expression
            else:
                expression = '\\text{H} ' + expression
            body = body.replace(TEX_TEXT_TO_REPLACE, expression)
        with open(result, "w") as outfile:
            outfile.write(body)
    return result

def tex_to_dvi(tex_file):
    result = tex_file.replace(".tex", ".dvi")
    if not os.path.exists(result):
        commands = [
            "latex",
            "-interaction=batchmode",
            "-halt-on-error",
            "-output-directory=" + TEX_DIR,
            tex_file  # ,
            # ">",
            # get_null()
        ]
        exit_code = os.system(" ".join(commands))
        if exit_code != 0:
            latex_output = ''
            log_file = tex_file.replace(".tex", ".log")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    latex_output = f.read()
            raise Exception(
                "Latex error converting to dvi. "
                "See log output above or the log file: %s" % log_file)
    return result

def dvi_to_svg(dvi_file):
    """
    Converts a dvi, which potentially has multiple slides, into a
    directory full of enumerated svgs corresponding with these slides.
    Returns a list of PIL Image objects for these images sorted as they
    where in the dvi
    """

    result = dvi_file.replace(".dvi", ".svg")
    result = result.replace("tex", "svg")  # change directory for the svg file
    print('svg: ', result)
    if not os.path.exists(result):
        commands = [
            "dvisvgm",
            dvi_file,
            "-n",
            "-v",
            "3",
            "-o",
            result
            # Not sure what these are for, and it seems to work without them
            # so commenting out for now
            # ,
            # ">",
            # get_null()
        ]
        os.system(" ".join(commands))
    return result

def hashed_tex(expression, typeface):
    string = expression + typeface
    hasher = hashlib.sha256(string.encode())
    return hasher.hexdigest()[:16]

#### Alignment functions

def get_figure_curves(imported_svg_data, fig):
    """
        returns the curves
        :param fig:
        :return:
        """

    return imported_svg_data[fig]['curves']

def calc_lengths(imported_svg_data):
    """
        The dimension of the object is calculated
        The parameters are stored in

        | imported_svg_data:
        | 'length'
        | 'height'
        | 'centerx'
        | 'centery'
        | 'beginning'
        | 'end'
        | 'top'
        | 'bottom'

        :return:
        """
    lengths = []
    for expr in imported_svg_data:
        curves = get_figure_curves(imported_svg_data, expr)  # the H is stripped for latex formulas

        max_vals = [-math.inf, -math.inf]
        min_vals = [math.inf, math.inf]

        directions = [0, 1]  # 0 horizontal
        # 1 vertical

        for direction in directions:
            for char in curves:
                for spline in char.data.splines:
                    for point in spline.bezier_points:
                        candidate = char.matrix_local.translation[direction] + point.co[direction] * char.scale[
                            direction]  # +char.parent.matrix_local.translation[direction]
                        if max_vals[direction] < candidate:
                            max_vals[direction] = candidate
                        if min_vals[direction] > candidate:
                            min_vals[direction] = candidate

        right_most_x, top_most_y = max_vals
        left_most_x, bottom_most_y = min_vals

        length = right_most_x - left_most_x
        center = left_most_x + length / 2

        imported_svg_data[expr]['length'] = length
        imported_svg_data[expr]['centerx'] = center
        imported_svg_data[expr]['beginning'] = left_most_x  # Untested
        imported_svg_data[expr]['end'] = right_most_x

        height = top_most_y - bottom_most_y
        center = bottom_most_y + height / 2

        imported_svg_data[expr]['top'] = top_most_y
        imported_svg_data[expr]['bottom'] = bottom_most_y
        imported_svg_data[expr]['height'] = height
        imported_svg_data[expr]['centery'] = center
        imported_svg_data[expr]['curves'] = curves
    return imported_svg_data

def align_figures(imported_svg_data, aligned):
    imported_svg_data = calc_lengths(imported_svg_data)
    for fig in imported_svg_data:
        imported_svg_data[fig] = align_figure(fig, imported_svg_data, aligned)
    return imported_svg_data

def align_figure(fig, imported_svg_data, aligned):
    data = imported_svg_data
    curve_list = data[fig]['curves']
    offset = list(curve_list[0].location)

    # positioning of the first element of the list
    if aligned == 'right':
        offset[0] = data[fig]['end']
        offset[1] = data[fig]['centery']
    if aligned == 'right_bottom':
        offset[0] = data[fig]['end']
        offset[1] = data[fig]['bottom']
    elif aligned == 'top_centered':
        offset[0] = data[fig]['centerx']
        offset[1] = data[fig]['top']
    elif aligned == 'left_bottom':
        offset[0] = data[fig]['beginning']
        offset[1] = data[fig]['bottom']
    elif aligned == 'left_top':
        offset[0] = data[fig]['beginning']
        offset[1] = data[fig]['top']
    elif aligned == 'x_and_y':
        offset[0] = data[fig]['centerx']
        offset[1] = data[fig]['centery']
    elif aligned == 'left':
        offset[0] = data[fig]['beginning']
        offset[1] = data[fig]['centery']
    elif aligned == 'center':
        cen = data[fig]['centerx']
        offset[0] = cen
        offset[1] = data[fig]['centery']

    for i in range(len(curve_list)):
        loc = list(curve_list[i].location)
        new_loc = add_lists_by_element(loc, offset, subtract=True)
        curve_list[i].location = new_loc

    # sort letters from left to right
    curve_list.sort(key=lambda x:x.location[0])
    # remove reference H
    return curve_list[1:]

def add_lists_by_element(list1, list2, subtract=False):
    if len(list1) != len(list2):
        raise Warning("The lists aren't the same length")
    list3 = list(deepcopy(list2))
    if subtract:
        for i in range(len(list3)):
            list3[i] *= -1
    return list(map(sum, zip(list1, list3)))
