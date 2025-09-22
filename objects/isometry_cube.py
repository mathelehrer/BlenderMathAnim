from math import pi

from interface import ibpy
from interface.ibpy import get_geometry_node_from_modifier, get_default_value_at_frame, Vector, Quaternion, Euler
from new_stuff.geometry_nodes_modifier import TransformationModifier, TransformationSphereModifier
from objects.text2 import MorphText2
from objects.bobject import BObject
from objects.cube import Cube
from objects.text import Text
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE
from utils.kwargs import get_from_kwargs

r2 = 1/2**0.5

class IsometryCube(BObject):
    def __init__(self, permutation=None,label=False,**kwargs):
        """
        a cube that is used for visualizing the isometries CoxB3
        the cube can appear as

        * a simple wireframe
        * with labelled vertices
        * with colored faces

        """
        rotation_euler=get_from_kwargs(kwargs,'rotation_euler',Euler((0,0,0)))
        cube = Cube()

        self.name = get_from_kwargs(kwargs, 'name', "IsometryCube")
        self.word = get_from_kwargs(kwargs, 'word', "")

        self.label_text = None
        self.scale=get_from_kwargs(kwargs,'scale',[1]*3)
        self.location = get_from_kwargs(kwargs,"location",Vector())

        if label:
            if permutation is None:
                self.label_text = Text(self.word, name=self.name + "Label",
                                       aligned="center",outline_emission=5,**kwargs)
            else:
                self.label_text = MorphText2(self.word, permutation, name=self.name + "Label",
                                             aligned="center",outline_emission=5,**kwargs)
            # make the label invisible, only the instance of the label will be visible
            ibpy.hide(self.label_text, begin_time=0)


        if self.label_text:
            self.trans_mod = TransformationModifier(name="TransformationModifier", font_material='joker',
                                               edge_material='plastic_example',
                                               face_material='six_color_ramp',
                                                    grid_material='plastic_custom1',
                                                    label=self.label_text.ref_obj,**kwargs)
        else:
            self.trans_mod = TransformationModifier(name="TransformationModifier", font_material='joker',
                                                    edge_material='plastic_example',
                                                    face_material='six_color_ramp',
                                                    grid_material='plastic_custom1', **kwargs)


        cube.add_mesh_modifier(type="NODES", node_modifier=self.trans_mod)
        self.rotation_counter = 0 # counter for added rotations
        self.last_global_rotation=Vector()
        self.last_label_rotatation=Vector()

        super().__init__(obj=cube.ref_obj,rotation_euler=rotation_euler, location=self.location,name=self.name, **kwargs)
        ibpy.set_origin(self,type='ORIGIN_GEOMETRY')
        if self.label_text:
            ibpy.set_parent(self.label_text,self)

    def grow(self,scale = None, begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        self.appear(alpha=0, begin_time=begin_time,
                    transition_time=0, silent=True,**kwargs)
        nice_alpha = get_from_kwargs(kwargs,'nice_alpha',True)
        if nice_alpha:
            ibpy.set_taa_render_samples(1024, begin_time * FRAME_RATE)
            ibpy.set_taa_render_samples(64, (begin_time + transition_time) * FRAME_RATE)
        self.appear_frame(begin_time=begin_time,transition_time=0)
        if scale is None:
            scale = self.intrinsic_scale
        ibpy.grow(self,scale,begin_time*FRAME_RATE,transition_time*FRAME_RATE)
        return begin_time+transition_time

    def write_label(self,reverse=True,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        self.label_text.write(reverse=reverse,begin_time=begin_time,transition_time=transition_time,**kwargs)
        return begin_time+transition_time

    def unwrite_label(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        self.label_text.unwrite( begin_time=begin_time, transition_time=transition_time, **kwargs)
        return begin_time + transition_time

    def morph_label(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        return self.label_text.morph(begin_time=begin_time,transition_time=transition_time)

    def unmorph_label(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        return self.label_text.unmorph(begin_time=begin_time,transition_time=transition_time)

    def appear_frame(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        edge_material = self.trans_mod.materials[0]
        ibpy.change_alpha_of_material(edge_material,from_value=0,to_value=1,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

    def appear_corner_labels(self,with_background=False,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        font_material = self.trans_mod.materials[1]
        ibpy.change_alpha_of_material(font_material,from_value=0,to_value=1,begin_time=begin_time,transition_time=transition_time)
        if with_background:
            font_back_material = self.trans_mod.materials[2]
            ibpy.change_alpha_of_material(font_back_material, from_value=0, to_value=1, begin_time=begin_time,
                                          transition_time=transition_time)
        return begin_time+transition_time

    def show_corner_label(self,idx = 1,begin_time=0):
        label_switch = ibpy.get_geometry_node_from_modifier(self.trans_mod,label="LabelSwitch")
        ibpy.change_default_value(label_switch.inputs[idx],from_value=0,to_value=1,begin_time=begin_time,transition_time=0)
        return begin_time

    def show_corner_labels(self,indices=[],begin_time=0):
        for idx in indices:
            self.show_corner_label(idx=idx,begin_time=begin_time)
        return begin_time

    def unshow_corner_labels(self,indices=[],begin_time=0):
        label_switch = ibpy.get_geometry_node_from_modifier(self.trans_mod, label="LabelSwitch")
        [ibpy.change_default_value(label_switch.inputs[idx], from_value=1, to_value=0, begin_time=begin_time,
                                  transition_time=0) for idx in indices]
        return begin_time

    def rotate_corner_labels(self,rotation_euler=[0,0,0],begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):

        rotation_node = ibpy.get_geometry_node_from_modifier(self.trans_mod, label="CornerRotation")
        ibpy.change_default_vector(rotation_node,from_value=self.last_label_rotatation,to_value=rotation_euler,
                                   begin_time=begin_time,transition_time=transition_time)
        self.last_corner_label_rotation = rotation_euler
        return begin_time+transition_time

    def disappear_corner_labels(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        font_material = self.trans_mod.materials[1]
        font_back_material = self.trans_mod.materials[2]
        ibpy.change_alpha_of_material(font_material,from_value=1,to_value=0,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

    def appear_face_colors(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        face_material = self.trans_mod.materials[3]
        ibpy.change_alpha_of_material(face_material,from_value=0,to_value=1,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

    def add_rotation(self,quaternion = Quaternion(),begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        quat_node= get_geometry_node_from_modifier(self.trans_mod,label="Rotation"+str(self.rotation_counter))
        ibpy.change_default_quaternion(quat_node,from_value=Quaternion(),to_value=quaternion,begin_time=begin_time,transition_time=transition_time)
        self.rotation_counter+=1
        return begin_time+transition_time

    def apply_word(self,word = None,simultaneous=True,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        reflection_nodes=[get_geometry_node_from_modifier(self.trans_mod,label="CoxeterReflection"+str(i)) for i in range(10)]
        reflection_normals=[get_geometry_node_from_modifier(self.trans_mod,label="Normal"+str(i)) for i in range(10)]
        if word is None:
            word = self.word

        if word == r"\varepsilon":
            # no transformation needed for the identity
            return begin_time

        # reverse word order (words are interpreted from right to left)
        word = word[::-1]

        for letter,normal in zip(word,reflection_normals):
            if letter=='a':
                ibpy.change_default_vector(normal,from_value=Vector((0,0,0)),to_value=Vector((0,0,1)),begin_time=-0.1,transition_time=0.1)
            elif letter=='b':
                ibpy.change_default_vector(normal,from_value=Vector((0,0,0)),to_value=Vector((r2,0,r2)),begin_time=-0.1,transition_time=0.1)
            elif letter=='c':
                ibpy.change_default_vector(normal,from_value=Vector((0,0,0)),to_value=Vector((r2,r2,0)),begin_time=-0.1,transition_time=0.1)

        if simultaneous:
            for reflection_node in reflection_nodes:
                ibpy.change_default_value(reflection_node.inputs["progress"],from_value=0,to_value=1,begin_time=begin_time,transition_time=transition_time)
        else:
            dt = transition_time/len(word)
            for i,reflection_node in enumerate(reflection_nodes[0:len(word)]):
                ibpy.change_default_value(reflection_node.inputs["progress"],from_value=0,to_value=1,begin_time=begin_time+i*dt,transition_time=dt)

        return begin_time+transition_time

    def apply_word_part(self,parts=[],word = None,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        reflection_nodes=[get_geometry_node_from_modifier(self.trans_mod,label="CoxeterReflection"+str(i)) for i in range(10)]
        reflection_normals=[get_geometry_node_from_modifier(self.trans_mod,label="Normal"+str(i)) for i in range(10)]
        if word is None:
            word = self.word

        for letter,normal in zip(word,reflection_normals):
            if letter=='a':
                ibpy.change_default_vector(normal,from_value=Vector((0,0,0)),to_value=Vector((0,0,1)),begin_time=0,transition_time=0)
            elif letter=='b':
                ibpy.change_default_vector(normal,from_value=Vector((0,0,0)),to_value=Vector((r2,0,r2)),begin_time=0,transition_time=0)
            elif letter=='c':
                ibpy.change_default_vector(normal,from_value=Vector((0,0,0)),to_value=Vector((r2,r2,0)),begin_time=0,transition_time=0)


        dt = transition_time/len(parts)
        for i in parts:
            reflection_node=reflection_nodes[i]
            ibpy.change_default_value(reflection_node.inputs["progress"],from_value=0,to_value=1,begin_time=begin_time+i*dt,transition_time=dt)

        self.label_text.write(begin_time=begin_time,transition_time=transition_time,from_letter=min(parts)+0.001,to_letter=max(parts)+1)
        return begin_time+transition_time


    def show_mirror(self,generator='a',begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        # get grid rotation node
        grid_material = self.trans_mod.materials[4]
        ibpy.change_alpha_of_material(grid_material, from_value=0, to_value=1, begin_time=begin_time,
                                      transition_time=transition_time)
        grid_rotation_node = get_geometry_node_from_modifier(self.trans_mod,label="GridRotation")
        if generator=='a':
            pass
        elif generator == 'b':
            ibpy.change_default_vector(grid_rotation_node,from_value=Vector((0,0,0)),to_value=Vector((0,pi/4,0)),begin_time=0,transition_time=0)
        elif generator == 'c':
            ibpy.change_default_vector(grid_rotation_node,from_value=Vector((0,0,0)),to_value=Vector((pi/2,0,-pi/4)),begin_time=0,transition_time=0)
        radius = get_geometry_node_from_modifier(self.trans_mod,label="Radius")
        ibpy.change_default_value(radius,from_value=0,to_value=0.01,begin_time=0,transition_time=transition_time)
        return begin_time + transition_time

    def shift_label(self,direction=Vector(),begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        return self.label_text.move(direction,begin_time=begin_time,transition_time=transition_time)


    def hide_mirror(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        color = self.trans_mod.materials[3]
        ibpy.change_alpha_of_material(color, from_value=1, to_value=0, begin_time=begin_time,
                                      transition_time=transition_time)
        return begin_time + transition_time

    def disappear(self,alpha=0,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        colors = self.trans_mod.materials
        sockets= [color.node_tree.nodes.get("Principled BSDF").inputs.get("Alpha") for color in colors]
        alphas = [get_default_value_at_frame(socket,begin_time*FRAME_RATE) for socket in sockets]

        [ibpy.change_default_value(socket,from_value=alpha_old,to_value=alpha,begin_time=begin_time,transition_time=transition_time) for socket,alpha_old in zip(sockets,alphas)]
        if alpha==0:
            ibpy.hide(self, begin_time=begin_time+transition_time)

        return begin_time+transition_time

    def rotate(self,rotation_euler=Vector(),begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        rotation_node= get_geometry_node_from_modifier(self.trans_mod,label="GlobalRotation")
        ibpy.change_default_vector(rotation_node,from_value=self.last_global_rotation,to_value=rotation_euler,begin_time=begin_time,transition_time=transition_time)
        self.last_global_rotation=rotation_euler
        return begin_time+transition_time

class IsometrySphere(BObject):
    def __init__(self, permutation=None,**kwargs):
        """
        a sphere that is used for visualizing the isometries CoxB3

        """
        rotation_euler=get_from_kwargs(kwargs,'rotation_euler',Euler((0,0,0)))
        cube = Cube()

        self.name = get_from_kwargs(kwargs, 'name', "IsometryCube")
        self.word = get_from_kwargs(kwargs, 'word', "")

        self.label_text = None
        self.scale=get_from_kwargs(kwargs,'scale',[1]*3)

        initial_state = get_from_kwargs(kwargs, 'initial_state', "wireframe")

        self.trans_mod = TransformationSphereModifier(name="TransformationSphereModifier",face_material='six_color_ramp', **kwargs)

        cube.add_mesh_modifier(type="NODES", node_modifier=self.trans_mod)
        self.rotation_counter = 0 # counter for added rotations
        self.last_global_rotation=Vector()
        super().__init__(obj=cube.ref_obj,rotation_euler=rotation_euler, name=self.name, **kwargs)
        ibpy.set_origin(self,type='ORIGIN_GEOMETRY')

    def grow(self,scale = None, begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        self.appear(alpha=0, begin_time=begin_time,
                    transition_time=0, silent=True,**kwargs)
        nice_alpha = get_from_kwargs(kwargs,'nice_alpha',True)
        if nice_alpha:
            ibpy.set_taa_render_samples(1024, begin_time * FRAME_RATE)
            ibpy.set_taa_render_samples(64, (begin_time + transition_time) * FRAME_RATE)
        self.appear(begin_time=begin_time,transition_time=0)
        if scale is None:
            scale = self.intrinsic_scale
        ibpy.grow(self,scale,begin_time*FRAME_RATE,transition_time*FRAME_RATE)
        return begin_time+transition_time

    def add_rotation(self,quaternion = Quaternion(),begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        quat_node= get_geometry_node_from_modifier(self.trans_mod,label="Rotation"+str(self.rotation_counter))
        ibpy.change_default_quaternion(quat_node,from_value=Quaternion(),to_value=quaternion,begin_time=begin_time,transition_time=transition_time)
        self.rotation_counter+=1
        return begin_time+transition_time

    def apply_word(self,word = None,simultaneous=True,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        reflection_nodes=[get_geometry_node_from_modifier(self.trans_mod,label="CoxeterReflection"+str(i)) for i in range(10)]
        reflection_normals=[get_geometry_node_from_modifier(self.trans_mod,label="Normal"+str(i)) for i in range(10)]
        if word is None:
            word = self.word

        if word == r"\varepsilon":
            # no transformation needed for the identity
            return begin_time

        # reverse word order (words are interpreted from right to left)
        word = word[::-1]

        for letter,normal in zip(word,reflection_normals):
            if letter=='a':
                ibpy.change_default_vector(normal,from_value=Vector((0,0,0)),to_value=Vector((0,0,1)),begin_time=-0.1,transition_time=0.1)
            elif letter=='b':
                ibpy.change_default_vector(normal,from_value=Vector((0,0,0)),to_value=Vector((r2,0,r2)),begin_time=-0.1,transition_time=0.1)
            elif letter=='c':
                ibpy.change_default_vector(normal,from_value=Vector((0,0,0)),to_value=Vector((r2,r2,0)),begin_time=-0.1,transition_time=0.1)

        if simultaneous:
            for reflection_node in reflection_nodes:
                ibpy.change_default_value(reflection_node.inputs["progress"],from_value=0,to_value=1,begin_time=begin_time,transition_time=transition_time)
        else:
            dt = transition_time/len(word)
            for i,reflection_node in enumerate(reflection_nodes[0:len(word)]):
                ibpy.change_default_value(reflection_node.inputs["progress"],from_value=0,to_value=1,begin_time=begin_time+i*dt,transition_time=dt)

        return begin_time+transition_time

    def apply_word_part(self,parts=[],word = None,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        reflection_nodes=[get_geometry_node_from_modifier(self.trans_mod,label="CoxeterReflection"+str(i)) for i in range(10)]
        reflection_normals=[get_geometry_node_from_modifier(self.trans_mod,label="Normal"+str(i)) for i in range(10)]
        if word is None:
            word = self.word

        for letter,normal in zip(word,reflection_normals):
            if letter=='a':
                ibpy.change_default_vector(normal,from_value=Vector((0,0,0)),to_value=Vector((0,0,1)),begin_time=0,transition_time=0)
            elif letter=='b':
                ibpy.change_default_vector(normal,from_value=Vector((0,0,0)),to_value=Vector((r2,0,r2)),begin_time=0,transition_time=0)
            elif letter=='c':
                ibpy.change_default_vector(normal,from_value=Vector((0,0,0)),to_value=Vector((r2,r2,0)),begin_time=0,transition_time=0)


        dt = transition_time/len(parts)
        for i in parts:
            reflection_node=reflection_nodes[i]
            ibpy.change_default_value(reflection_node.inputs["progress"],from_value=0,to_value=1,begin_time=begin_time+i*dt,transition_time=dt)

        self.label_text.write(begin_time=begin_time,transition_time=transition_time,from_letter=min(parts)+0.001,to_letter=max(parts)+1)
        return begin_time+transition_time

    def disappear(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        colors = self.trans_mod.materials
        sockets= [color.node_tree.nodes.get("Principled BSDF").inputs.get("Alpha") for color in colors]
        alphas = [get_default_value_at_frame(socket,begin_time*FRAME_RATE) for socket in sockets]

        [ibpy.change_default_value(socket,from_value=alpha,to_value=0,begin_time=begin_time,transition_time=transition_time) for socket,alpha in zip(sockets,alphas)]
        ibpy.hide(self, begin_time=begin_time+transition_time)

        return begin_time+transition_time

    def rotate(self,rotation_euler=Vector(),begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        rotation_node= get_geometry_node_from_modifier(self.trans_mod,label="GlobalRotation")
        ibpy.change_default_vector(rotation_node,from_value=self.last_global_rotation,to_value=rotation_euler,begin_time=begin_time,transition_time=transition_time)
        self.last_global_rotation=rotation_euler
        return begin_time+transition_time
