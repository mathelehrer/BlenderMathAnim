import numpy as np
from anytree import NodeMixin
from mathutils import Vector

from objects.bmatrix import BMatrix
from objects.bobject import BObject
from objects.curve import Curve
from objects.cylinder import Cylinder
from objects.tex_bobject import TexBObject
from utils.constants import OBJECT_APPEARANCE_TIME


class Node(BObject, NodeMixin):
    """
    A node of a tree that consists of
    * a label
    * a box
    * a connection to the parent
    """

    def __init__(self, element, parent=None, to_string=str, action='right',**kwargs):
        self.action= action
        self.kwargs = kwargs
        self.current_mapping = None  # if there is more than one expression in the node
        self.center = None
        self.line = None
        self.a = 1
        self.b = 1
        self.element = element
        self.thickness = self.get_from_kwargs('thickness', 0.01)
        self.bevel = self.get_from_kwargs('bevel',0)# grap bevel for the font
        self.color_function = self.get_from_kwargs('color_function', lambda x: ['drawing'])
        self.name = self.get_from_kwargs('name', 'Node')
        self.node_circles=self.get_from_kwargs('node_circles',True)
        self.display_mode = self.get_from_kwargs('display_mode', 'word')
        self.to_string=to_string

        # create bezier curve around the expression
        # read out location before children are created
        self.location = self.get_from_kwargs('location', [0, 0, 0])
        scale = self.get_from_kwargs('scale', 1)

        if not isinstance(self.location, Vector):
            self.location = Vector(self.location)

        expressions = []
        colors=[]
        if hasattr(element,"shortest_word"):
            shortest_word=  element.shortest_word
        else:
            shortest_word = str(element)
        if str(element) == shortest_word:
            expressions.append(str(element))
            colors.append(self.color_function(str(element)))
        else:
            expressions.append(str(element))
            expressions.append(shortest_word)
            colors.append(self.color_function(str(element)))
            colors.append(self.color_function(shortest_word))

        if self.display_mode=='word':
            self.texBObject = TexBObject(*expressions, scale=scale, colors=colors, name=self.name + "_tex",
                                         aligned='center', bevel=self.bevel,**kwargs)
        else:
            self.texBObject = BMatrix(element.matrix,name=self.name+"_tex",aligned='center',bevel=self.bevel,**element.kwargs)

        mappings = []
        if isinstance(self.texBObject,BMatrix):
            [x_min,y_min,z_min,x_max,y_max,z_max]=self.texBObject.get_text_bounding_box()
            xm = 0.5 * (x_min + x_max)
            a = (x_max - x_min)
            ym = 0.5 * (y_min + y_max)
            b = (y_max - y_min)
            a *= scale
            b *= scale
            self.a = a
            self.b = b
        else:
            for i, obj in enumerate(self.texBObject.objects):
                [x_min, y_min, z_min, x_max,y_max, z_max] = obj.get_text_bounding_box()
                xm = 0.5 * (x_min + x_max)
                a = (x_max - x_min)
                ym = 0.5 * (y_min + y_max)
                b = (y_max - y_min)
                a *= scale
                b *= scale
                if i == 0:
                    self.a = a
                    self.b = b
                mappings.append(lambda phi, a=a: [a * np.cos(phi) + xm, 0, b * np.sin(phi) + ym])

        self.center = Vector([xm + self.location.x, 0, ym + self.location.z])
        if self.node_circles:
            self.bFrame = Curve(mappings, domain=[0, 2 * np.pi], color='text', name='curve_' + self.name,
                            thickness=self.thickness, **kwargs)

        if parent:
            c_p = parent.center  # center parent
            c_c = self.center  # center child
            diff = c_c - c_p
            # calculate intersection with elliptical frame of parent
            # if a modified scalar product is introduced <u,v> =<b*b*u.x*v.x+a*a*u.z*v.z>, the affine parameter of the intersection
            # is calculated with l = +- a*b/sqrt(<d,d>)
            # with our choice of direction the intersection uses the postive value and is given by c_p+l*d
            d = np.sqrt(parent.a ** 2 * diff.z ** 2 + parent.b ** 2 * diff.x ** 2)
            l = parent.a * parent.b / d
            intersection_p = c_p + diff * l

            # repeat for the child node
            # here the negative solution has to be taken
            d = np.sqrt(self.a ** 2 * diff.z ** 2 + self.b ** 2 * diff.x ** 2)
            l = self.a * self.b / d
            intersection_c = c_c - diff * l

            if action=='right':
                color = self.color_function(str(element))[-1]
            else:
                color = self.color_function(str(element))[0]

            self.line = Cylinder.from_start_to_end(end=intersection_c, start=intersection_p,
                                                   color=color,
                                                   name='connection_to_' + self.name,
                                                   thickness=0.5 * self.thickness,
                                                   **kwargs)  # the line is not part of the node, it is a child of the tree
            children=[self.texBObject]
            if hasattr(self,'bFrame'):
                children.append(self.bFrame)
            super().__init__(children=children, name=self.name+"_Node_" + expressions[0],
                             location=self.location, color='text', **kwargs)
            self.parent = parent
        else:
            self.line = None
            children=[self.texBObject]
            if hasattr(self,'bFrame'):
                children.append(self.bFrame)
            super().__init__(children=children, name=self.name,
                             location=self.location, **kwargs)


    def __str__(self):
        return self.to_string(self.element)

    def appear(self,
               begin_time=0,
               transition_time=0,
               **kwargs):
        super().appear(begin_time=begin_time, transition_time=transition_time)
        if isinstance(self.texBObject,TexBObject):
            self.texBObject.write_index(0, begin_time=begin_time + transition_time / 4,transition_time=transition_time / 4 * 3)
        else:
            self.texBObject.write(begin_time=begin_time+transition_time/4,transition_time=transition_time/4*3)
        if hasattr(self,'bFrame'):
            self.bFrame.grow(begin_time=begin_time + transition_time / 4, transition_time=transition_time / 2)
        return begin_time+transition_time

    def next(self, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
        """
        morph to next node, provided that there is a shorter word available for self
        :param begin_time:
        :param transition_time:
        :return:
        """

        self.texBObject.next(begin_time=begin_time, transition_time=transition_time)
        self.texBObject.perform_morphing()
        if self.bFrame:
            self.bFrame.next(begin_time=begin_time, transition_time=transition_time)

        count = 1
        for child in self.children:
            child.next(begin_time=begin_time + 0.1 * count * transition_time,
                       transition_time=transition_time - 0.1 * count * transition_time)
            count += 1
        return begin_time+transition_time
    
