import hashlib
import math
import os
from copy import deepcopy

import numpy as np

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
    def __init__(self,expression,sample_points=101, **kwargs):
        """
        example:
        text = Text("Hallo Welt",color="drawing",outline_color="example",aligned="center",
        emission=0.5,outline_emission=2)
        text.write(begin_time=1,transition_time=1)


        """
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name',"TextObject")
        self.rotation = self.get_from_kwargs('rotation', Vector((math.pi / 2, 0, 0)))
        self.location = self.get_from_kwargs('location', Vector((0, 0, 0)))

        self.modifier = TextModifier(expression,rotation=self.rotation,location=self.location,sample_points=sample_points,**kwargs)
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
        outline_radius = get_from_kwargs(kwargs,'outline_radius',0.01)

        material_node.inputs['Material'].default_value= mat
        outline_material_node.inputs['Material'].default_value = mat_outline

        keep_outline = get_from_kwargs(kwargs, 'keep_outline', False)
        keep_outline_node = get_geometry_node_from_modifier(self.modifier, "KeepOutline")
        if keep_outline:
            ibpy.change_default_boolean(keep_outline_node,False,True,begin_frame=0)
        else:
            ibpy.change_default_boolean(keep_outline_node,True,False,begin_frame=0)
        outline_radius_node = get_geometry_node_from_modifier(self.modifier, "OutlineRadius")
        if outline_radius!=0.01:
            ibpy.change_default_value(outline_radius_node,from_value=0.01,to_value=outline_radius,begin_frame=0)

        super().__init__(obj=cube, name=self.name, no_material=True, **kwargs)
        super().add_mesh_modifier('NODES', node_modifier=self.modifier)

    def write(self,from_letter=0,to_letter=None,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        reverse=get_from_kwargs(kwargs,"reverse",False)
        if reverse:
            reverse_control=get_geometry_node_from_modifier(self.modifier,"ReverseControl")
            ibpy.change_default_boolean(reverse_control,from_value=False,to_value=True,begin_time=begin_time)
        else:
            reverse_control=get_geometry_node_from_modifier(self.modifier,"ReverseControl")
            ibpy.change_default_boolean(reverse_control,from_value=False,to_value=False,begin_time=begin_time)
        if to_letter is None:
            to_letter = self.modifier.number_of_letters
        write_control = get_geometry_node_from_modifier(self.modifier,"WriteControlNode")
        ibpy.change_default_value(write_control,from_value=from_letter,to_value=to_letter,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

    def disappear(self, alpha=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        return self.unwrite(begin_time=begin_time, transition_time=transition_time, **kwargs)

    def unwrite(self,letters=None,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):

        all_letters = self.modifier.number_of_letters
        if letters is None:
            letters = all_letters
        write_control = get_geometry_node_from_modifier(self.modifier,"WriteControlNode")
        ibpy.change_default_value(write_control,from_value=all_letters,to_value=all_letters-letters,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

    def length(self):
        return self.modifier.number_of_letters

    def get_text_bounding_box(self):
        '''
        designated for SimpleTexBOjects
        it works on the level of letters
        :param texBObject:
        :return:
        '''
        x_min = np.Infinity
        x_max = -np.Infinity
        y_min = np.Infinity
        y_max = -np.Infinity
        z_min = np.Infinity
        z_max = -np.Infinity
        obj = self.ref_obj
        loc = obj.location
        bb = obj.bound_box  # eight corner coordinates of the surrounding box
        for b in bb:
            if (b[0] + loc[0]) < x_min:
                x_min = b[0] + loc[0]
            if (b[0] + loc[0]) > x_max:
                x_max = b[0] + loc[0]
            if (b[1] + loc[1]) < y_min:
                y_min = b[1] + loc[1]
            if (b[1] + loc[1]) > y_max:
                y_max = b[1] + loc[1]
            if (b[2] + loc[2]) < z_min:
                z_min = b[2] + loc[2]
            if (b[2] + loc[2]) > z_max:
                z_max = b[2] + loc[2]
        return [x_min, y_min, z_min, x_max, y_max, z_max]


class MorphText(BObject):
    """
    A new class for a morphing text object based on geometry nodes
    """
    def __init__(self,expression1,expression2,morph_shift=Vector(),sample_points=101, **kwargs):
        """
        example:
        text = MorphText(r"\text{Hallo Welt}",\text{"Welcome!"},sample_points=1001,color="drawing",outline_color="example",aligned="center",
        emission=0.5,outline_emission=2)
        t0  = 1
        t0 = 0.5+ text.write(begin_time=1,transition_time=1)
        t0 = 0.5 + text.morph(begin_time=t0,transition_time=1)

        """
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name',"TextObject")
        self.rotation = self.get_from_kwargs('rotation', Vector((math.pi / 2, 0, 0)))
        self.location = self.get_from_kwargs('location', Vector((0, 0, 0)))

        self.modifier = MorphTextModifier(expression1,expression2,morph_shift=morph_shift,sample_points=sample_points,
                                          rotation=self.rotation,location=self.location,**kwargs)
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

    def write(self,from_letter=0,to_letter=None,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        if to_letter is None:
            to_letter = self.modifier.number_of_letters
        write_control = get_geometry_node_from_modifier(self.modifier,"WriteControlNode")
        ibpy.change_default_value(write_control,from_value=from_letter,to_value=to_letter,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

    def unwrite(self,letters,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        all_letters = self.modifier.number_of_letters
        write_control = get_geometry_node_from_modifier(self.modifier,"WriteControlNode")
        ibpy.change_default_value(write_control,from_value=all_letters,to_value=all_letters-letters,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

    def morph(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        morph_control = get_geometry_node_from_modifier(self.modifier,"MorphControlNode")
        ibpy.change_default_value(morph_control,from_value=0,to_value=1,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

    def unwrite(self, letters=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        all_letters = self.modifier.number_of_letters
        write_control = get_geometry_node_from_modifier(self.modifier, "WriteControlNode")
        ibpy.change_default_value(write_control, from_value=all_letters, to_value=all_letters - letters,
                                  begin_time=begin_time,
                                  transition_time=transition_time)
        return begin_time + transition_time

def below(out,buffer_y=200):
    return out.location-Vector((0,buffer_y))

class TextModifier(GeometryNodesModifier):
    def __init__(self,expression,**kwargs):
        self.expression = expression
        self.number_of_letters =0
        self.sample_points = get_from_kwargs(kwargs,'sample_points',101)
        self.group_outputs = None
        super().__init__(get_from_kwargs(kwargs,'name',"GeoText"),
                         group_input=False,group_output=False,automatic_layout=False,**kwargs)


    def create_node(self,tree,**kwargs):

        self.number_of_letters = generate_expression(self.expression, **kwargs)
        create_from_xml(tree,"geo_fonts",**kwargs)

        self.group_outputs = tree.nodes.get("GroupOutput")

        collection_info = tree.nodes.get("TextData")
        collection_info.inputs["Separate Children"].default_value = True
        # blender cuts the collection names to a length of 63 letter
        # in order to find the collection in the node setup, the name must be shortened appropriately
        # TODO If a name only differs after the first 63 letters, the proper collection might not be found

        collection_name = hashed_tex(self.expression)
        # print("abbreviated collection name: ",collection_name)
        collection_info.inputs["Collection"].default_value = get_collection(collection_name)

        # adjust parameter
        sample_points_nodes = tree.nodes.get("SamplePointsNode")
        sample_points_nodes.integer = self.sample_points

        for n in tree.nodes:
            if n.label=='FontMaterial':
                material_node=n
            if n.label=='OutlineMaterial':
                outline_material_node=n
            if n.label=='LastJoin':
                last_join_node=n
            if n.label=="Out":
                out=n

        material = get_material(get_from_kwargs(kwargs,'color',"text"),**kwargs)
        self.materials.append(material)
        material_node.inputs['Material'].default_value= material

        outline_emission = get_from_kwargs(kwargs,'emission_outline',1)
        kwargs['emission']=outline_emission
        outline_material = get_material(get_from_kwargs(kwargs,'outline_color',"text"),**kwargs)
        self.materials.append(outline_material)
        outline_material_node.inputs['Material'].default_value = outline_material

        expr_location = InputVector(tree, name="ExprLocation",
                                     value=get_from_kwargs(kwargs,'location',Vector()))
        expr_rotation = InputVector(tree, name="ExprRotation",value=get_from_kwargs(kwargs,'rotation',Vector()))


        transform_geometry = TransformGeometry(tree,name="ExprTransform",translation=expr_location.std_out,
                                               rotation=expr_rotation.std_out)

        # override default location
        transform_geometry.node.location=below(out)
        expr_location.node.location=below(transform_geometry.node,buffer_y=400)
        expr_rotation.node.location=below(expr_location.node,buffer_y=200)

        tree.links.new(last_join_node.outputs["Geometry"],transform_geometry.geometry_in)
        tree.links.new(transform_geometry.geometry_out,out.inputs['Geometry'])


class MorphTextModifier(GeometryNodesModifier):
    def __init__(self,expression1,expression2,**kwargs):
        self.expression1 = expression1
        self.expression2 = expression2
        self.number_of_letters =0
        self.morph_shift = get_from_kwargs(kwargs,'morph_shift',Vector())
        self.sample_points = get_from_kwargs(kwargs,'sample_points',101)

        super().__init__(get_from_kwargs(kwargs,'name',"GeoText"),
                         group_input=False,group_output=False,automatic_layout=False,**kwargs)

    def create_node(self,tree,**kwargs):

        self.number_of_letters = generate_expression(self.expression1, **kwargs)
        self.number_of_morph_letters = generate_expression(self.expression2, **kwargs)

        create_from_xml(tree,"geo_morph_fonts",**kwargs)

        collection_info = tree.nodes.get("TextData")
        collection_info.inputs["Separate Children"].default_value = True

        collection_info2 = tree.nodes.get("MorphData")
        collection_info2.inputs["Separate Children"].default_value = True

        # blender cuts the collection names to a length of 63 letter
        # in order to find the collection in the node setup, the name must be shortened appropriately
        # TODO If a name only differs after the first 63 letters, the proper collection might not be found

        collection_name1 = hashed_tex(self.expression1)
        # print("abbreviated collection name: ",collection_name)
        collection_info.inputs["Collection"].default_value = get_collection(collection_name1)

        collection_name2 = hashed_tex(self.expression2)
        collection_info2.inputs["Collection"].default_value = get_collection(collection_name2)

        # adjust parameter
        sample_points_nodes = tree.nodes.get("SamplePointsNode")
        sample_points_nodes.integer=self.sample_points

        morph_shift_nodes = tree.nodes.get("MorphShiftNode")
        morph_shift_nodes.vector=self.morph_shift

        for n in tree.nodes:
            if n.label=='FontMaterial':
                material_node=n
            if n.label=='OutlineMaterial':
                outline_material_node=n
            if n.label=='LastJoin':
                last_join_node=n
            if n.label=="Out":
                out=n

        material = get_material(get_from_kwargs(kwargs,'color',"text"),**kwargs)
        self.materials.append(material)
        material_node.inputs['Material'].default_value= material

        outline_emission = get_from_kwargs(kwargs,'emission_outline',1)
        kwargs['emission']=outline_emission
        outline_material = get_material(get_from_kwargs(kwargs,'outline_color',"text"),**kwargs)
        self.materials.append(outline_material)
        outline_material_node.inputs['Material'].default_value = outline_material

        expr_location = InputVector(tree, name="ExprLocation",
                                     value=get_from_kwargs(kwargs,'location',Vector()))
        expr_rotation = InputVector(tree, name="ExprRotation",value=get_from_kwargs(kwargs,'rotation',Vector()))


        transform_geometry = TransformGeometry(tree,name="ExprTransform",translation=expr_location.std_out,
                                               rotation=expr_rotation.std_out)

        # override default location
        transform_geometry.node.location=below(out)
        expr_location.node.location=below(transform_geometry.node,buffer_y=400)
        expr_rotation.node.location=below(expr_location.node,buffer_y=200)

        tree.links.new(last_join_node.outputs["Geometry"],transform_geometry.geometry_in)
        tree.links.new(transform_geometry.geometry_out,out.inputs['Geometry'])


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
        collection = ibpy.make_new_collection(hashed_tex(expression),hide_render=True,hide_viewport=True)
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

def hashed_tex(expression, typeface="default"):
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
