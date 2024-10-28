import os
import random

import bpy
import numpy as np
from sympy.parsing.mathematica import parse_mathematica

from geometry_nodes.nodes import make_function, Switch
from interface.ibpy import make_new_socket
from shader_nodes.shader_nodes import MixNode
from utils.constants import LOC_FILE_DIR, DATA_DIR
from utils.kwargs import get_from_kwargs

def fitting_function_dictionary():
    with open(os.path.join(DATA_DIR, "fittingfunctions.dat")) as data:
        raw = ""
        for line in data:
            raw += line

        raw = raw.replace("<|", "")
        raw = raw.replace("|>", "")
        raw = raw.replace(r"*^", "e")
        raw = raw.replace("^","**")
        parts = raw.split(",")
        dict = {}
        for part in parts:
            sub_parts = part.split("->")
            dict[int(sub_parts[0])] =sub_parts[1]
        return dict

class GenericNode:
    def __init__(self,tree,location=(0,0),**kwargs):
        """
        The generic node can be used as a ShaderNode and
        GeometryNode, it automatically detects the correct
        type from the tree type that is provided in the arguments
        """
        if isinstance(tree,bpy.types.GeometryNodeTree):
            self.node_group_type='GeometryNodes'
        else:
            self.node_group_type='Shader'
        self.tree = tree
        self.links = tree.links
        self.location =location

        self.links = tree.links
        self.nodes = tree.nodes

        hide = get_from_kwargs(kwargs,"hide",True)
        self.hide = hide


class LegendrePolynomial(GenericNode):
    def __init__(self,tree,l=3,location=(0,0),x=0,**kwargs):
        """
        a collection of function group nodes are combined into one node
        The values of the Legendre polynomial are computed recursively,
        to reduce numerical inaccuracies that arise from evaluations of large powers
        in the polynomials. The hope is that the recursive computation is numerically more stable
        """
        self.l = l
        self.name=get_from_kwargs(kwargs,'name',"LegendreP"+str(l))
        super().__init__(tree,location=location,**kwargs)

        # create the correct node group
        if 'Geometry' in self.node_group_type:
            sub_tree = bpy.data.node_groups.new(type='GeometryNodeTree',name=self.name)
            group = tree.nodes.new(type='GeometryNodeGroup')
        else:
            sub_tree=bpy.data.node_groups.new(type='ShaderNodeTree',name=self.name)
            group = tree.nodes.new(type='ShaderNodeGroup')

        group.location=(location[0]*200,location[1]*100)
        group.name=self.name
        group.label=self.name
        group.node_tree = sub_tree

        # create the sockets for the group
        group_inputs = sub_tree.nodes.new('NodeGroupInput')
        group_outputs = sub_tree.nodes.new('NodeGroupOutput')
        make_new_socket(sub_tree,name="x",io='INPUT',type='NodeSocketFloat')
        make_new_socket(sub_tree,name=self.name,io='OUTPUT',type='NodeSocketFloat')

        group_inputs.location=(-(0.5*l+1)*200,0)
        group_outputs.location=(0,0)

        left = -0.5*l

        p0 = make_function(sub_tree.nodes,
            functions={
                "P0":"1"
            },
            location =(left,0),
            inputs=["x"],outputs=["P0"],
            scalars=["x","P0"],
            node_group_type=self.node_group_type,
            name="P0",
        )
        ps = [p0]
        if l>0:
            p1 = make_function(sub_tree.nodes,
                functions={
                    "P1":"x"
                },
                location =(left,1),
                inputs=["x"],outputs=["P1"],
                scalars=["x","P1"],
                node_group_type=self.node_group_type,
                name="P1",
            )
            ps.append(p1)

        for l in range(2,l+1):
            expr = "2,"+str(l)+",*,1,-,x,*,P"+str(l-1)+",*,"+str(l-1)+",P"+str(l-2)+",*,-,"+str(l)+",/"
            p_next = make_function(sub_tree.nodes,
                                   functions={
                                       "P"+str(l):expr
                                   },
                                   location=(left+0.5*(l-1),  l%2),
                                   inputs=["x","P"+str(l-1),"P"+str(l-2)], outputs=["P"+str(l)],
                                   scalars=["x", "P"+str(l),"P"+str(l-1),"P"+str(l-2)],
                                   node_group_type=self.node_group_type,
                                   name="P"+str(l),
                    )
            sub_tree.links.new(ps[-1].outputs[0],p_next.inputs["P"+str(l-1)])
            sub_tree.links.new(ps[-2].outputs[0],p_next.inputs["P"+str(l-2)])
            ps.append(p_next)

        for p in ps:
            sub_tree.links.new(group_inputs.outputs[0],p.inputs["x"])

        sub_tree.links.new(ps[-1].outputs[0],group_outputs.inputs[0])
        self.node = group

        if isinstance(x, (float, int)):
            self.node.inputs["x"].default_value = x
        else:
            tree.links.new(x, self.node.inputs["x"])

        self.std_out = self.node.outputs[0]


class AssociatedLegendrePolynomial(GenericNode):
    def __init__(self,tree,l=3,m=2,location=(0,0),x=0,y=0,**kwargs):
        """
        a collection of function group nodes are combined into one node
        The values of the associated Legendre polynomial are computed recursively,
        to reduce numerical inaccuracies that arise from evaluations of large powers
        in the polynomials. The hope is that the recursive computation is numerically more stable

        the inputs are as follows
        x: cos(theta)
        y: sin(theta)
        """
        self.l = l
        self.m = m
        self.name=get_from_kwargs(kwargs,'name',"AssociatedLegendreP"+str(l)+"_"+str(m))
        super().__init__(tree,location=location,**kwargs)

        # create the correct node group
        if 'Geometry' in self.node_group_type:
            sub_tree = bpy.data.node_groups.new(type='GeometryNodeTree',name=self.name)
            group = tree.nodes.new(type='GeometryNodeGroup')
        else:
            sub_tree=bpy.data.node_groups.new(type='ShaderNodeTree',name=self.name)
            group = tree.nodes.new(type='ShaderNodeGroup')

        group.location=(location[0]*200,location[1]*100)
        group.name=self.name
        group.label=self.name
        group.node_tree = sub_tree

        # create the sockets for the group
        group_inputs = sub_tree.nodes.new('NodeGroupInput')
        group_outputs = sub_tree.nodes.new('NodeGroupOutput')
        make_new_socket(sub_tree,name="x",io='INPUT',type='NodeSocketFloat')
        make_new_socket(sub_tree,name="y",io='INPUT',type='NodeSocketFloat')
        make_new_socket(sub_tree,name=self.name,io='OUTPUT',type='NodeSocketFloat')

        group_inputs.location=(-(0.85*l+1+abs(m))*200,0)
        group_outputs.location=(0,0)

        left = -0.85*l-abs(m)
        # make Legendre polynomials
        p0 = make_function(sub_tree.nodes,
            functions={
                "P0":"1"
            },
            location =(left,0),
            inputs=["x"],outputs=["P0"],
            scalars=["x","P0"],
            node_group_type=self.node_group_type,
            name="P0",
        )
        ps = [p0]
        if l>0:
            p1 = make_function(sub_tree.nodes,
                functions={
                    "P1":"x"
                },
                location =(left,1),
                inputs=["x"],outputs=["P1"],
                scalars=["x","P1"],
                node_group_type=self.node_group_type,
                name="P1",
            )
            ps.append(p1)

        for l in range(2,l+1):
            expr = "2,"+str(l)+",*,1,-,x,*,P"+str(l-1)+",*,"+str(l-1)+",P"+str(l-2)+",*,-,"+str(l)+",/"
            p_next = make_function(sub_tree.nodes,
                                   functions={
                                       "P"+str(l):expr
                                   },
                                   location=(left+0.85*(l-1),  l%2),
                                   inputs=["x","P"+str(l-1),"P"+str(l-2)], outputs=["P"+str(l)],
                                   scalars=["x", "P"+str(l),"P"+str(l-1),"P"+str(l-2)],
                                   node_group_type=self.node_group_type,
                                   name="P"+str(l),
                    )
            sub_tree.links.new(ps[-1].outputs[0],p_next.inputs["P"+str(l-1)])
            sub_tree.links.new(ps[-2].outputs[0],p_next.inputs["P"+str(l-2)])
            ps.append(p_next)

        for p in ps:
            sub_tree.links.new(group_inputs.outputs[0],p.inputs["x"])

        sub_tree.links.new(ps[-1].outputs[0],group_outputs.inputs[0])

        # make derivatives

        left = -0.85 * l-abs(m)
        # make Legendre polynomials
        dp0 = make_function(sub_tree.nodes,
                           functions={
                               "P'0": "0"
                           },
                           location=(left, -2),
                           inputs=["x"], outputs=["P'0"],
                           scalars=["x", "P'0"],
                           node_group_type=self.node_group_type,
                           name="P'0",
                           )
        dps = [dp0]
        if l > 0:
            dp1 = make_function(sub_tree.nodes,
                               functions={
                                   "P'1": "1"
                               },
                               location=(left, -1),
                               inputs=["x"], outputs=["P'1"],
                               scalars=["x", "P'1"],
                               node_group_type=self.node_group_type,
                               name="P'1",
                               )
            dps.append(dp1)

        for l in range(2, l + 1):
            expr = "2," + str(l) + ",*,1,-,P" + str(l - 1) + ",x,P'"+str(l-1)+",*,+,*," + str(l - 1) + ",P'" + str(l - 2) + ",*,-," + str(
                l) + ",/"
            dp_next = make_function(sub_tree.nodes,
                                   functions={
                                       "P'" + str(l): expr
                                   },
                                   location=(left + 0.85 * (l - 1),-2+ l % 2),
                                   inputs=["x","P"+str(l-1), "P'" + str(l - 1), "P'" + str(l - 2)],
                                    outputs=["P'" + str(l)],
                                   scalars=["x", "P"+str(l-1),"P'" + str(l), "P'" + str(l - 1), "P'" + str(l - 2)],

                                   node_group_type=self.node_group_type,
                                   name="P'" + str(l),
                                   )
            sub_tree.links.new(ps[l-1].outputs[0], dp_next.inputs["P" + str(l - 1)])
            sub_tree.links.new(dps[-1].outputs[0], dp_next.inputs["P'" + str(l - 1)])
            sub_tree.links.new(dps[-2].outputs[0], dp_next.inputs["P'" + str(l - 2)])
            dps.append(dp_next)

        for dp in dps:
            sub_tree.links.new(group_inputs.outputs[0], dp.inputs["x"])
        sub_tree.links.new(dps[-1].outputs[0], group_outputs.inputs[0])


        left = -abs(m)
        # make associated Legendre functions
        # use the following recursive relation P_l^{m} = -(l+m-1)(l-m+2)P_{l}^{m-2}-2(m-1)*x/y*P_l^{m-1}
        al0 = make_function(sub_tree.nodes,
            functions={
                "P"+str(l)+"_0":"P"+str(l)
            },
            location =(left,-4),
            inputs=["x","y","P"+str(l)],outputs=["P"+str(l)+"_0"],
            scalars=["x","y","P"+str(l),"P"+str(l)+"_0"],
            node_group_type=self.node_group_type,
            name="P"+str(l)+"_0",
        )
        sub_tree.links.new(ps[l].outputs[0], al0.inputs["P"+str(l)])
        als = [al0]

        if m>0:
            al1 = make_function(sub_tree.nodes,
                                functions={
                                    "P"+str(l)+"_1":"-1,y,*,P'"+str(l)+",*"
                },
                location =(left,-3),
                inputs=["x","y","P"+str(l),"P'"+str(l)],outputs=["P"+str(l)+"_1"],
                scalars=["x","y","P"+str(l),"P'"+str(l),"P"+str(l)+"_1"],
                node_group_type=self.node_group_type,
                name="P"+str(l)+"_1")
            sub_tree.links.new(ps[l].outputs[0], al1.inputs["P" + str(l)])
            sub_tree.links.new(dps[l].outputs[0], al1.inputs["P'" + str(l)])
        else:
            al1 = make_function(sub_tree.nodes,
                                functions={
                                    "P" + str(l) + "_-1": "y,P'" + str(l) + ",*,"+str(l*(l+1))+",/"
                                },
                                location=(left, -3),
                                inputs=["x", "y", "P" + str(l), "P'" + str(l)], outputs=["P" + str(l) + "_-1"],
                                scalars=["x", "y", "P" + str(l), "P'" + str(l), "P" + str(l) + "_-1"],
                                node_group_type=self.node_group_type,
                                name="P" + str(l) + "_-1")
            sub_tree.links.new(ps[l].outputs[0], al1.inputs["P" + str(l)])
            sub_tree.links.new(dps[l].outputs[0], al1.inputs["P'" + str(l)])

        als.append(al1)

        # use the following recursive relation P_l^{m} = -(l+m-1)(l-m+2)P_{l}^{m-2}-2(m-1)*x/y*P_l^{m-1}
        if m>0:
            for m in range(2,m+1):
                expr = str(-(l+m-1)*(l-m+2))+",P"+str(l)+"_"+str(m-2)+",*,"+str(2*(m-1))+",x,*,y,/,P"+str(l)+"_"+str(m-1)+",*,-"
                al_next = make_function(sub_tree.nodes,
                                        functions={
                                            "P" + str(l)+"_"+str(m): expr
                                        },
                                        location=(left + 0.85 * (m - 1), -2 + m % 2),
                                        inputs=["x","y", "P" + str(l)+"_"+str(m-1),"P"+str(l)+"_"+str(m-2)],
                                        outputs=["P" + str(l)+"_"+str(m)],
                                        scalars=["x","y", "P" + str(l)+"_"+str(m-1), "P" + str(l)+"_"+str(m-2),"P" + str(l)+"_"+str(m)],

                                        node_group_type=self.node_group_type,
                                        name="P" + str(l)+"_"+str(m),
                                        )
                sub_tree.links.new(als[-1].outputs[0], al_next.inputs["P" + str(l)+"_"+str(m-1)])
                sub_tree.links.new(als[-2].outputs[0], al_next.inputs["P" + str(l)+"_"+str(m-2)])
                als.append(al_next)
        else:
            for m in range(2, abs(m) + 1):
                expr = "2,"+str(m-1)+",*,x,*,y,/,P"+str(l)+"_"+str(-m+1)+",*,P"+str(l)+"_"+str(-m+2)+",-,"+str((l-m+1)*(l+m))+",/"
                al_next = make_function(sub_tree.nodes,
                                        functions={
                                            "P" + str(l) + "_" + str(-m): expr
                                        },
                                        location=(left + 0.85 * (m - 1), -2 + m % 2),
                                        inputs=["x", "y", "P" + str(l) + "_" + str(-m+1),
                                                "P" + str(l) + "_" + str(-m +2)],
                                        outputs=["P" + str(l) + "_" + str(-m)],
                                        scalars=["x", "y", "P" + str(l) + "_" + str(-m + 1),
                                                 "P" + str(l) + "_" + str(-m + 2), "P" + str(l) + "_" + str(-m)],

                                        node_group_type=self.node_group_type,
                                        name="P" + str(l) + "_" + str(-m),
                                        )
                sub_tree.links.new(als[-1].outputs[0], al_next.inputs["P" + str(l) + "_" + str(-m + 1)])
                sub_tree.links.new(als[-2].outputs[0], al_next.inputs["P" + str(l) + "_" + str(-m + 2)])
                als.append(al_next)

        for al in als:
            sub_tree.links.new(group_inputs.outputs[0], al.inputs["x"])
            sub_tree.links.new(group_inputs.outputs[1], al.inputs["y"])

        if m==0:
            sub_tree.links.new(als[-2].outputs[0], group_outputs.inputs[0])
        else:
            sub_tree.links.new(als[-1].outputs[0], group_outputs.inputs[0])

        # finalizing things
        self.node = group
        if isinstance(x, (float, int)):
            self.node.inputs["x"].default_value = x
        else:
            tree.links.new(x, self.node.inputs["x"])

        if isinstance(y, (float, int)):
            self.node.inputs["y"].default_value = y
        else:
            tree.links.new(y, self.node.inputs["y"])
        self.std_out = self.node.outputs[0]


class SphericalHarmonicsRekursive(GenericNode):
    def __init__(self,tree,l=3,m=2,location=(0,0),theta=0,phi=0,**kwargs):
        """
        a collection of function group nodes are combined into one node
        The values of the spherical harmonics are computed recursively,
        to reduce numerical inaccuracies that arise from evaluations of large powers
        in the polynomials. The hope is that the recursive computation is numerically more stable

        the inputs are as follows
        x: cos(theta)
        y: sin(theta)
        """
        self.l = l
        self.m = m
        self.name=get_from_kwargs(kwargs,'name',"Y"+str(l)+"_"+str(m))
        super().__init__(tree,location=location,**kwargs)

        # create the correct node group
        if 'Geometry' in self.node_group_type:
            sub_tree = bpy.data.node_groups.new(type='GeometryNodeTree',name=self.name)
            group = tree.nodes.new(type='GeometryNodeGroup')
        else:
            sub_tree=bpy.data.node_groups.new(type='ShaderNodeTree',name=self.name)
            group = tree.nodes.new(type='ShaderNodeGroup')

        group.location=(location[0]*200,location[1]*100)
        group.name=self.name
        group.label=self.name
        group.hide =self.hide
        group.node_tree = sub_tree

        # create the sockets for the group
        group_inputs = sub_tree.nodes.new('NodeGroupInput')
        group_outputs = sub_tree.nodes.new('NodeGroupOutput')
        make_new_socket(sub_tree,name="theta",io='INPUT',type='NodeSocketFloat')
        make_new_socket(sub_tree,name="phi",io='INPUT',type='NodeSocketFloat')
        make_new_socket(sub_tree,name=self.name+"_re",io='OUTPUT',type='NodeSocketFloat')
        make_new_socket(sub_tree,name=self.name+"_im",io='OUTPUT',type='NodeSocketFloat')

        group_inputs.location=(-(0.85*l+2+abs(m))*200,0)
        group_outputs.location=(0,0)

        left = -(0.85 * l+1 + abs(m))

        xy = make_function(sub_tree.nodes,
                           functions={
                               "x":"theta,cos",
                               "y": "theta,sin"
                           },
                           name="xyTransform",location=(left,0),
                           scalars=["x","y","theta"],inputs=["theta"],outputs=["x","y"],
                           node_group_type=self.node_group_type)

        sub_tree.links.new(group_inputs.outputs["theta"],xy.inputs["theta"])
        x_socket = xy.outputs["x"]
        y_socket = xy.outputs["y"]

        m=abs(self.m)
        factor = 1
        for i in range(-m+1,m+1):
            factor=factor*np.sqrt(self.l+i)
            if i>0:
                factor/=2*i
        factor*=np.sqrt((2*self.l+1) / 4 / np.pi)

        sign_neg = 1
        if self.m<0:
            sign_neg *=(-1)**(self.l+self.m)
        else:
            sign_neg *=(-1)**self.l
        sign_pos = 1
        if self.m>=0:
            sign_pos *=(-1)**self.m
        approx = make_function(sub_tree, name="Asymptotics",
                                    functions={
                                        "approx":str(factor)+",y,"+str(abs(self.m))+",**,*,"+str(sign_pos)+",x,0,>,*,"+str(sign_neg)+",x,0,<,*,+,*"
                                    }, inputs=["x","y"], outputs=["approx"],
                                    scalars=["x","y","approx"],
                                    location= (left,3),node_group_type=self.node_group_type)
        sub_tree.links.new(x_socket,approx.inputs["x"])
        sub_tree.links.new(y_socket,approx.inputs["y"])

        left = -0.85*l-abs(m)
        # make Legendre polynomials
        p0 = make_function(sub_tree.nodes,
            functions={
                "P0":"1"
            },
            location =(left,0),
            inputs=["x"],outputs=["P0"],
            scalars=["x","P0"],
            node_group_type=self.node_group_type,
            name="P0",
        )
        ps = [p0]
        if l>0:
            p1 = make_function(sub_tree.nodes,
                functions={
                    "P1":"x"
                },
                location =(left,1),
                inputs=["x"],outputs=["P1"],
                scalars=["x","P1"],
                node_group_type=self.node_group_type,
                name="P1",
            )
            ps.append(p1)

        for l in range(2,l+1):
            expr = "2,"+str(l)+",*,1,-,x,*,P"+str(l-1)+",*,"+str(l-1)+",P"+str(l-2)+",*,-,"+str(l)+",/"
            p_next = make_function(sub_tree.nodes,
                                   functions={
                                       "P"+str(l):expr
                                   },
                                   location=(left+0.85*(l-1),  l%2),
                                   inputs=["x","P"+str(l-1),"P"+str(l-2)], outputs=["P"+str(l)],
                                   scalars=["x", "P"+str(l),"P"+str(l-1),"P"+str(l-2)],
                                   node_group_type=self.node_group_type,
                                   name="P"+str(l),
                    )
            sub_tree.links.new(ps[-1].outputs[0],p_next.inputs["P"+str(l-1)])
            sub_tree.links.new(ps[-2].outputs[0],p_next.inputs["P"+str(l-2)])
            ps.append(p_next)

        for p in ps:
            sub_tree.links.new(x_socket,p.inputs["x"])

        sub_tree.links.new(ps[-1].outputs[0],group_outputs.inputs[0])

        # make derivatives

        left = -0.85 * l-abs(m)
        # make Legendre polynomials
        dp0 = make_function(sub_tree.nodes,
                           functions={
                               "P'0": "0"
                           },
                           location=(left, -2),
                           inputs=["x"], outputs=["P'0"],
                           scalars=["x", "P'0"],
                           node_group_type=self.node_group_type,
                           name="P'0",
                           )
        dps = [dp0]
        if l > 0:
            dp1 = make_function(sub_tree.nodes,
                               functions={
                                   "P'1": "1"
                               },
                               location=(left, -1),
                               inputs=["x"], outputs=["P'1"],
                               scalars=["x", "P'1"],
                               node_group_type=self.node_group_type,
                               name="P'1",
                               )
            dps.append(dp1)

        for l in range(2, l + 1):
            expr = "2," + str(l) + ",*,1,-,P" + str(l - 1) + ",x,P'"+str(l-1)+",*,+,*," + str(l - 1) + ",P'" + str(l - 2) + ",*,-," + str(
                l) + ",/"
            dp_next = make_function(sub_tree.nodes,
                                   functions={
                                       "P'" + str(l): expr
                                   },
                                   location=(left + 0.85 * (l - 1),-2+ l % 2),
                                   inputs=["x","P"+str(l-1), "P'" + str(l - 1), "P'" + str(l - 2)],
                                    outputs=["P'" + str(l)],
                                   scalars=["x", "P"+str(l-1),"P'" + str(l), "P'" + str(l - 1), "P'" + str(l - 2)],

                                   node_group_type=self.node_group_type,
                                   name="P'" + str(l),
                                   )
            sub_tree.links.new(ps[l-1].outputs[0], dp_next.inputs["P" + str(l - 1)])
            sub_tree.links.new(dps[-1].outputs[0], dp_next.inputs["P'" + str(l - 1)])
            sub_tree.links.new(dps[-2].outputs[0], dp_next.inputs["P'" + str(l - 2)])
            dps.append(dp_next)

        for dp in dps:
            sub_tree.links.new(x_socket, dp.inputs["x"])
        sub_tree.links.new(dps[-1].outputs[0], group_outputs.inputs[0])


        left = -abs(m)
        # make associated Legendre functions
        # use the following recursive relation P_l^{m} = -(l+m-1)(l-m+2)P_{l}^{m-2}-2(m-1)*x/y*P_l^{m-1}
        al0 = make_function(sub_tree.nodes,
            functions={
                "P"+str(l)+"_0":"P"+str(l)
            },
            location =(left,-4),
            inputs=["x","y","P"+str(l)],outputs=["P"+str(l)+"_0"],
            scalars=["x","y","P"+str(l),"P"+str(l)+"_0"],
            node_group_type=self.node_group_type,
            name="P"+str(l)+"_0",
        )
        sub_tree.links.new(ps[l].outputs[0], al0.inputs["P"+str(l)])
        als = [al0]

        if m>0:
            al1 = make_function(sub_tree.nodes,
                                functions={
                                    "P"+str(l)+"_1":"-1,y,*,P'"+str(l)+",*,"+str(l*(l+1))+",sqrt,/"
                },
                location =(left,-3),
                inputs=["x","y","P"+str(l),"P'"+str(l)],outputs=["P"+str(l)+"_1"],
                scalars=["x","y","P"+str(l),"P'"+str(l),"P"+str(l)+"_1"],
                node_group_type=self.node_group_type,
                name="P"+str(l)+"_1")
            sub_tree.links.new(ps[l].outputs[0], al1.inputs["P" + str(l)])
            sub_tree.links.new(dps[l].outputs[0], al1.inputs["P'" + str(l)])
        else:
            # differs only by a sign from the one for positive m
            al1 = make_function(sub_tree.nodes,
                                functions={
                                    "P" + str(l) + "_-1": "y,P'" + str(l) + ",*,"+str(l*(l+1))+",sqrt,/"
                                },
                                location=(left, -3),
                                inputs=["x", "y", "P" + str(l), "P'" + str(l)], outputs=["P" + str(l) + "_-1"],
                                scalars=["x", "y", "P" + str(l), "P'" + str(l), "P" + str(l) + "_-1"],
                                node_group_type=self.node_group_type,
                                name="P" + str(l) + "_-1")
            sub_tree.links.new(ps[l].outputs[0], al1.inputs["P" + str(l)])
            sub_tree.links.new(dps[l].outputs[0], al1.inputs["P'" + str(l)])

        als.append(al1)

        # use the following recursive relation P_l^{m} = -(l+m-1)(l-m+2)P_{l}^{m-2}-2(m-1)*x/y*P_l^{m-1}
        if m>0:
            for m in range(2,m+1):
                expr = str((l+m-1)*(l-m+2))+","+str((l+m)*(l-m+1))+",/,sqrt"+",P"+str(l)+"_"+str(m-2)+",*,-1,*,"+str(2*(m-1))+","+str((l+m)*(l-m+1))+",sqrt,/,x,*,y,/,P"+str(l)+"_"+str(m-1)+",*,-"
                al_next = make_function(sub_tree.nodes,
                                        functions={
                                            "P" + str(l)+"_"+str(m): expr
                                        },
                                        location=(left + 0.85 * (m - 1), -2 + m % 2),
                                        inputs=["x","y", "P" + str(l)+"_"+str(m-1),"P"+str(l)+"_"+str(m-2)],
                                        outputs=["P" + str(l)+"_"+str(m)],
                                        scalars=["x","y", "P" + str(l)+"_"+str(m-1), "P" + str(l)+"_"+str(m-2),"P" + str(l)+"_"+str(m)],

                                        node_group_type=self.node_group_type,
                                        name="P" + str(l)+"_"+str(m),
                                        )
                sub_tree.links.new(als[-1].outputs[0], al_next.inputs["P" + str(l)+"_"+str(m-1)])
                sub_tree.links.new(als[-2].outputs[0], al_next.inputs["P" + str(l)+"_"+str(m-2)])
                als.append(al_next)
        else:
            for m in range(2, abs(m) + 1):
                # only one sign different
                expr = str((l + m - 1) * (l - m + 2)) + "," + str((l + m) * (l - m + 1)) + ",/,sqrt" + ",P" + str(
                    l) + "_" + str(-m + 2) + ",*,-1,*," + str(2 * (m - 1)) + "," + str(
                    (l + m) * (l - m + 1)) + ",sqrt,/,x,*,y,/,P" + str(l) + "_" + str(-m + 1) + ",*,+"
                al_next = make_function(sub_tree.nodes,
                                        functions={
                                            "P" + str(l) + "_" + str(-m): expr
                                        },
                                        location=(left + 0.85 * (m - 1), -2 + m % 2),
                                        inputs=["x", "y", "P" + str(l) + "_" + str(-m+1),
                                                "P" + str(l) + "_" + str(-m +2)],
                                        outputs=["P" + str(l) + "_" + str(-m)],
                                        scalars=["x", "y", "P" + str(l) + "_" + str(-m + 1),
                                                 "P" + str(l) + "_" + str(-m + 2), "P" + str(l) + "_" + str(-m)],

                                        node_group_type=self.node_group_type,
                                        name="P" + str(l) + "_" + str(-m),
                                        )
                sub_tree.links.new(als[-1].outputs[0], al_next.inputs["P" + str(l) + "_" + str(-m + 1)])
                sub_tree.links.new(als[-2].outputs[0], al_next.inputs["P" + str(l) + "_" + str(-m + 2)])
                als.append(al_next)

        for al in als:
            sub_tree.links.new(x_socket, al.inputs["x"])
            sub_tree.links.new(y_socket, al.inputs["y"])


        # finalizing things
        if self.m == 0:
            alp = als[-2]
        else:
            alp = als[-1]

        left+=1
        approximation = make_function(sub_tree, name="YlmAsymptotics",
                                      functions={
                                          "re": "approx," + str(self.m) + ",phi,*,cos,*",
                                          "im": "approx," + str(self.m) + ",phi,*,sin,*"
                                      }, inputs=["approx", "phi"], outputs=["re", "im"],
                                      scalars=["re", "im", "approx", "phi"], location=(left-2, 3),
                                      node_group_type=self.node_group_type)
        sub_tree.links.new(approx.outputs["approx"],approximation.inputs["approx"])
        sub_tree.links.new(group_inputs.outputs["phi"],approximation.inputs["phi"])
        left+=1
        # include phi, and rescaling by
        output = make_function(sub_tree,name="Y_lm",
                    functions={
                        "re":"alp,"+str(self.m)+",phi,*,cos,*,2,/,"+str(2*l+1)+",pi,/,sqrt,*,-1,"+str(self.m)+",**,*",
                        "im":"alp,"+str(self.m)+",phi,*,sin,*,2,/,"+str(2*l+1)+",pi,/,sqrt,*,-1,"+str(self.m)+",**,*"
                    },inputs=["alp","phi"],outputs=["re","im"],
                    scalars=["alp","phi","re","im"],location=(left,0),
                               node_group_type=self.node_group_type)
        sub_tree.links.new(alp.outputs[0],output.inputs["alp"])
        sub_tree.links.new(group_inputs.outputs["phi"],output.inputs["phi"])

        left+=1
        switch_function = make_function(sub_tree, name="switchFunction",
                                        functions={
                                            "switch": "approx,abs,0.001,<"
                                        }, inputs=["approx"], outputs=["switch"],
                                        scalars=["approx", "switch"], location=(left , 3),
                                        node_group_type=self.node_group_type)
        sub_tree.links.new(approx.outputs["approx"],switch_function.inputs["approx"])

        left+=1
        if self.node_group_type == 'Shader':
            mix_re = MixNode(sub_tree, factor=switch_function.outputs["switch"],
                             caseB=approximation.outputs["re"],
                             caseA=output.outputs["re"],
                             location=(left,3)
                             )
            mix_im = MixNode(sub_tree, factor=switch_function.outputs["switch"],
                             caseB=approximation.outputs["im"],
                             caseA=output.outputs["im"],
                             location=(left,1)
                             )

            sub_tree.links.new(mix_re.std_out, group_outputs.inputs[0])
            sub_tree.links.new(mix_im.std_out, group_outputs.inputs[1])
        else:
            mix_re = Switch(sub_tree,switch=switch_function.outputs["switch"],
                                false=output.outputs["re"],
                                true=approximation.outputs["re"],
                            input_type="FLOAT",
                                location=(left,3))
            mix_im = Switch(sub_tree, switch=switch_function.outputs["switch"],
                                false=output.outputs["im"],
                                true=approximation.outputs["im"],
                            input_type="FLOAT",
                                location=(left, 1))

            sub_tree.links.new(mix_re.outputs[0],group_outputs.inputs[0])
            sub_tree.links.new(mix_im.outputs[0],group_outputs.inputs[1])

        self.node = group
        if isinstance(theta, (float, int)):
            self.node.inputs["theta"].default_value = theta
        else:
            tree.links.new(theta, self.node.inputs["theta"])

        if isinstance(phi, (float, int)):
            self.node.inputs["phi"].default_value = phi
        else:
            tree.links.new(phi, self.node.inputs["phi"])

        self.phi = self.node.inputs["phi"]
        self.theta = self.node.inputs["theta"]
        self.re = self.node.outputs[self.name+"_re"]
        self.im = self.node.outputs[self.name+"_im"]

class SphericalHarmonics200(GenericNode):
    def __init__(self, tree,  ms=[2,5], ns = [2,5], location=(0, 0), theta=0, phi=0, **kwargs):
        """
        This is focused on the spherical harmonics with l=200,
        there are two parameter lists, one for the reals parts of the spherical harmonics and one for the imaginary part

        """

        l = 200
        self.name = get_from_kwargs(kwargs, 'name', "Y" + str(l) + "_" + str(ms))
        super().__init__(tree, location=location, **kwargs)

        # create the correct node group
        if 'Geometry' in self.node_group_type:
            sub_tree = bpy.data.node_groups.new(type='GeometryNodeTree', name=self.name)
            group = tree.nodes.new(type='GeometryNodeGroup')
        else:
            sub_tree = bpy.data.node_groups.new(type='ShaderNodeTree', name=self.name)
            group = tree.nodes.new(type='ShaderNodeGroup')

        group.location = (location[0] * 200, location[1] * 100)
        group.name = self.name
        group.label = self.name
        group.node_tree = sub_tree

        # create the sockets for the group
        group_inputs = sub_tree.nodes.new('NodeGroupInput')
        group_outputs = sub_tree.nodes.new('NodeGroupOutput')
        make_new_socket(sub_tree, name="theta", io='INPUT', type='NodeSocketFloat')
        make_new_socket(sub_tree, name="phi", io='INPUT', type='NodeSocketFloat')
        make_new_socket(sub_tree, name="Y", io='OUTPUT', type='NodeSocketFloat')
        max_m = 0
        for mabs in ms:
            if abs(mabs) > max_m:
                max_m = abs(mabs)
        for mabs in ns:
            if abs(mabs) > max_m:
                max_m = abs(mabs)

        group_inputs.location = (-(0.85 * l + 2 + max_m) * 200, 0)
        group_outputs.location = (0, 0)

        left = -(0.85 * l + 1 + max_m)

        xy = make_function(sub_tree.nodes,
                           functions={
                               "x": "theta,cos",
                               "y": "theta,sin"
                           },
                           name="xyTransform", location=(left, 0),
                           scalars=["x", "y", "theta"], inputs=["theta"], outputs=["x", "y"],
                           node_group_type=self.node_group_type)

        sub_tree.links.new(group_inputs.outputs["theta"], xy.inputs["theta"])
        x_socket = xy.outputs["x"]

        left = -0.85 * l - max_m

        left += 1
        zero = make_function(sub_tree, name="complexZero",
                                      functions={
                                          "re": "0",
                                          "im": "0"
                                      }, outputs=["re", "im"],
                                      scalars=["re", "im"], location=(left , 5),
                                      node_group_type=self.node_group_type)

        # the following nodes are independent of m and are shared between
        # associated Legendre polynomials of different m
        # make Legendre polynomials
        p0 = make_function(sub_tree.nodes,
                           functions={
                               "P0": "1"
                           },
                           location=(left, 0),
                           inputs=["x"], outputs=["P0"],
                           scalars=["x", "P0"],
                           node_group_type=self.node_group_type,
                           name="P0",
                           )
        ps = [p0]
        if l > 0:
            p1 = make_function(sub_tree.nodes,
                               functions={
                                   "P1": "x"
                               },
                               location=(left, 1),
                               inputs=["x"], outputs=["P1"],
                               scalars=["x", "P1"],
                               node_group_type=self.node_group_type,
                               name="P1",
                               )
            ps.append(p1)

        for l in range(2, l + 1):
            expr = "2," + str(l) + ",*,1,-,x,*,P" + str(l - 1) + ",*," + str(l - 1) + ",P" + str(l - 2) + ",*,-," + str(
                l) + ",/"
            p_next = make_function(sub_tree.nodes,
                                   functions={
                                       "P" + str(l): expr
                                   },
                                   location=(left + 0.85 * (l - 1), l % 2),
                                   inputs=["x", "P" + str(l - 1), "P" + str(l - 2)], outputs=["P" + str(l)],
                                   scalars=["x", "P" + str(l), "P" + str(l - 1), "P" + str(l - 2)],
                                   node_group_type=self.node_group_type,
                                   name="P" + str(l),
                                   )
            sub_tree.links.new(ps[-1].outputs[0], p_next.inputs["P" + str(l - 1)])
            sub_tree.links.new(ps[-2].outputs[0], p_next.inputs["P" + str(l - 2)])
            ps.append(p_next)

        for p in ps:
            sub_tree.links.new(x_socket, p.inputs["x"])

        sub_tree.links.new(ps[-1].outputs[0], group_outputs.inputs[0])

        # make derivatives

        left = -0.85 * l - max_m
        # make Legendre polynomials
        dp0 = make_function(sub_tree.nodes,
                            functions={
                                "P'0": "0"
                            },
                            location=(left, -2),
                            inputs=["x"], outputs=["P'0"],
                            scalars=["x", "P'0"],
                            node_group_type=self.node_group_type,
                            name="P'0",
                            )
        dps = [dp0]
        if l > 0:
            dp1 = make_function(sub_tree.nodes,
                                functions={
                                    "P'1": "1"
                                },
                                location=(left, -1),
                                inputs=["x"], outputs=["P'1"],
                                scalars=["x", "P'1"],
                                node_group_type=self.node_group_type,
                                name="P'1",
                                )
            dps.append(dp1)

        for l in range(2, l + 1):
            expr = "2," + str(l) + ",*,1,-,P" + str(l - 1) + ",x,P'" + str(l - 1) + ",*,+,*," + str(
                l - 1) + ",P'" + str(l - 2) + ",*,-," + str(
                l) + ",/"
            dp_next = make_function(sub_tree.nodes,
                                    functions={
                                        "P'" + str(l): expr
                                    },
                                    location=(left + 0.85 * (l - 1), -2 + l % 2),
                                    inputs=["x", "P" + str(l - 1), "P'" + str(l - 1), "P'" + str(l - 2)],
                                    outputs=["P'" + str(l)],
                                    scalars=["x", "P" + str(l - 1), "P'" + str(l), "P'" + str(l - 1),
                                             "P'" + str(l - 2)],

                                    node_group_type=self.node_group_type,
                                    name="P'" + str(l),
                                    )
            sub_tree.links.new(ps[l - 1].outputs[0], dp_next.inputs["P" + str(l - 1)])
            sub_tree.links.new(dps[-1].outputs[0], dp_next.inputs["P'" + str(l - 1)])
            sub_tree.links.new(dps[-2].outputs[0], dp_next.inputs["P'" + str(l - 2)])
            dps.append(dp_next)

        for dp in dps:
            sub_tree.links.new(x_socket, dp.inputs["x"])
        sub_tree.links.new(dps[-1].outputs[0], group_outputs.inputs[0])

        # also the first associated Legendre Polynomial is independent of m
        left = -mabs+5
        al0 = make_function(sub_tree.nodes,
                            functions={
                                "P" + str(l) + "_0": "P" + str(l)
                            },
                            location=(left, -4),
                            inputs=["x", "y", "P" + str(l)], outputs=["P" + str(l) + "_0"],
                            scalars=["x", "y", "P" + str(l), "P" + str(l) + "_0"],
                            node_group_type=self.node_group_type,
                            name="P" + str(l) + "_0",
                            )
        sub_tree.links.new(ps[l].outputs[0], al0.inputs["P" + str(l)])

        res=[]
        ims=[]
        coefficient_labels = []
        coefficient_sockets = []
        Y_labels = []

        realpart = []
        for m in ms:
            realpart.append(True)
        for n in ns:
            realpart.append(False)

        ms = ms+ns
        # here comes the m-dependent part
        # create all the real values

        for count,(current_m,real) in enumerate(zip(ms,realpart)):
            if real:
                label = "a_" + str(current_m)
            else:
                label = "b_" + str(current_m)

            coefficient_labels.append(label)
            make_new_socket(sub_tree, name=label, io='INPUT', type='NodeSocketFloat')
            coefficient_sockets.append(group_inputs.outputs[label])
            mabs=abs(current_m)

            # this is a fitting function that returns zero in a given range of domain, where the true computation
            # would be spoiled by numerical inaccuracies
            limit = lambda m: 0.000203063*m**2 - 4.19808e-6*m**3 + 4.3452e-8*m**4 - 2.12677e-10*m**5 + 3.95243e-13*m**6
            lim = limit(mabs)
            print("limit for zero: ",lim)

            switch= make_function(sub_tree, name="zeroRangeFor"+str(current_m),
                                   functions={
                                       "zero": "theta,"+str(lim)+",<,pi,theta,-,"+str(lim)+",<,+"
                                   }, inputs=["theta"], outputs=["zero"],
                                   scalars=["theta", "zero"],
                                   location=(left, -3-5*count), node_group_type=self.node_group_type)
            sub_tree.links.new(group_inputs.outputs["theta"], switch.inputs["theta"])

            y_function = make_function(sub_tree,name="ySocketFor"+str(current_m),
                        functions={
                            "y":"1,zero,1,=,*,theta,sin,zero,0,=,*,+"
                        },inputs=["theta","zero"],outputs=["y"],location=(left+0.5,-3.5-5*count),
                        scalars=["theta","zero","y"], node_group_type=self.node_group_type)
            sub_tree.links.new(group_inputs.outputs["theta"],y_function.inputs["theta"])
            sub_tree.links.new(switch.outputs["zero"],y_function.inputs["zero"])
            y_socket = y_function.outputs["y"]

            als_m = [al0]
            # make associated Legendre functions
            # use the following recursive relation P_l^{m} = -(l+m-1)(l-m+2)P_{l}^{m-2}-2(m-1)*x/y*P_l^{m-1}
            if current_m > 0:
                al1 = make_function(sub_tree.nodes,
                                    functions={
                                        "P" + str(l) + "_1": "-1,y,*,P'" + str(l) + ",*," + str(
                                            l * (l + 1)) + ",sqrt,/"
                                    },
                                    location=(left, -5-5*count),
                                    inputs=["x", "y", "P" + str(l), "P'" + str(l)], outputs=["P" + str(l) + "_1"],
                                    scalars=["x", "y", "P" + str(l), "P'" + str(l), "P" + str(l) + "_1"],
                                    node_group_type=self.node_group_type,
                                    name="P" + str(l) + "_1")
                sub_tree.links.new(ps[l].outputs[0], al1.inputs["P" + str(l)])
                sub_tree.links.new(dps[l].outputs[0], al1.inputs["P'" + str(l)])
            else:
                # differs only by a sign from the one for positive m
                al1 = make_function(sub_tree.nodes,
                                    functions={
                                        "P" + str(l) + "_-1": "y,P'" + str(l) + ",*," + str(l * (l + 1)) + ",sqrt,/"
                                    },
                                    location=(left, -5-5*count),
                                    inputs=["x", "y", "P" + str(l), "P'" + str(l)], outputs=["P" + str(l) + "_-1"],
                                    scalars=["x", "y", "P" + str(l), "P'" + str(l), "P" + str(l) + "_-1"],
                                    node_group_type=self.node_group_type,
                                    name="P" + str(l) + "_-1")
                sub_tree.links.new(ps[l].outputs[0], al1.inputs["P" + str(l)])
                sub_tree.links.new(dps[l].outputs[0], al1.inputs["P'" + str(l)])

            als_m.append(al1)

            # use the following recursive relation P_l^{m} = -(l+m-1)(l-m+2)P_{l}^{m-2}-2(m-1)*x/y*P_l^{m-1}
            if current_m > 0:
                for m in range(2, mabs + 1):
                    expr = str((l + m - 1) * (l - m + 2)) + "," + str(
                        (l + m) * (l - m + 1)) + ",/,sqrt" + ",P" + str(l) + "_" + str(m - 2) + ",*,-1,*," + str(
                        2 * (m - 1)) + "," + str((l + m) * (l - m + 1)) + ",sqrt,/,x,*,y,/,P" + str(l) + "_" + str(
                        m - 1) + ",*,-"
                    al_next = make_function(sub_tree.nodes,
                                            functions={
                                                "P" + str(l) + "_" + str(m): expr
                                            },
                                            location=(left + 0.85 * (m - 1), -5 + m % 2-5*count),
                                            inputs=["x", "y", "P" + str(l) + "_" + str(m - 1),
                                                    "P" + str(l) + "_" + str(m - 2)],
                                            outputs=["P" + str(l) + "_" + str(m)],
                                            scalars=["x", "y", "P" + str(l) + "_" + str(m - 1),
                                                     "P" + str(l) + "_" + str(m - 2), "P" + str(l) + "_" + str(m)],

                                            node_group_type=self.node_group_type,
                                            name="P" + str(l) + "_" + str(m),
                                            )
                    sub_tree.links.new(als_m[-1].outputs[0], al_next.inputs["P" + str(l) + "_" + str(m - 1)])
                    sub_tree.links.new(als_m[-2].outputs[0], al_next.inputs["P" + str(l) + "_" + str(m - 2)])
                    als_m.append(al_next)
            else:
                for m in range(2, mabs + 1):
                    # only one sign different
                    expr = str((l + m - 1) * (l - m + 2)) + "," + str(
                        (l + m) * (l - m + 1)) + ",/,sqrt" + ",P" + str(
                        l) + "_" + str(-m + 2) + ",*,-1,*," + str(2 * (m - 1)) + "," + str(
                        (l + m) * (l - m + 1)) + ",sqrt,/,x,*,y,/,P" + str(l) + "_" + str(-m + 1) + ",*,+"
                    al_next = make_function(sub_tree.nodes,
                                            functions={
                                                "P" + str(l) + "_" + str(-m): expr
                                            },
                                            location=(left + 0.85 * (m - 1), -5 + m % 2-5*count),
                                            inputs=["x", "y", "P" + str(l) + "_" + str(-m + 1),
                                                    "P" + str(l) + "_" + str(-m + 2)],
                                            outputs=["P" + str(l) + "_" + str(-m)],
                                            scalars=["x", "y", "P" + str(l) + "_" + str(-m + 1),
                                                     "P" + str(l) + "_" + str(-m + 2),
                                                     "P" + str(l) + "_" + str(-m)],

                                            node_group_type=self.node_group_type,
                                            name="P" + str(l) + "_" + str(-m),
                                            )
                    sub_tree.links.new(als_m[-1].outputs[0], al_next.inputs["P" + str(l) + "_" + str(-m + 1)])
                    sub_tree.links.new(als_m[-2].outputs[0], al_next.inputs["P" + str(l) + "_" + str(-m + 2)])
                    als_m.append(al_next)

            for al in als_m:
                sub_tree.links.new(x_socket, al.inputs["x"])
                sub_tree.links.new(y_socket, al.inputs["y"])

            # finalizing things
            if current_m == 0:
                alp = als_m[-2]
            else:
                alp = als_m[-1]

            left += 1
            # include phi, and rescaling by
            output = make_function(sub_tree, name="Y_lm",
                                   functions={
                                       "re": "alp," + str(current_m) + ",phi,*,cos,*,2,/," + str(
                                           2 * l + 1) + ",pi,/,sqrt,*,-1," + str(current_m) + ",**,*",
                                       "im": "alp," + str(current_m) + ",phi,*,sin,*,2,/," + str(
                                           2 * l + 1) + ",pi,/,sqrt,*,-1," + str(current_m) + ",**,*"
                                   }, inputs=["alp", "phi"], outputs=["re", "im"],
                                   scalars=["alp", "phi", "re", "im"], location=(left, 0.5-5*count),
                                   node_group_type=self.node_group_type,hide=True)
            sub_tree.links.new(alp.outputs[0], output.inputs["alp"])
            sub_tree.links.new(group_inputs.outputs["phi"], output.inputs["phi"])
            Y_labels.append("Y"+str(l)+"_"+str(current_m))

            left += 1
            if self.node_group_type == 'Shader':
                mix_re = MixNode(sub_tree, factor=switch.outputs["zero"],
                                 caseB=zero.outputs["re"],
                                 caseA=output.outputs["re"],
                                 location=(left, 1.5-5*count)
                                 )
                mix_im = MixNode(sub_tree, factor=switch.outputs["zero"],
                                 caseB=zero.outputs["im"],
                                 caseA=output.outputs["im"],
                                 location=(left, 1-5*count))
                sub_tree.links.new(mix_re.std_out, group_outputs.inputs[0])
                sub_tree.links.new(mix_im.std_out, group_outputs.inputs[1])
                res.append(mix_re)
                ims.append(mix_im)
            else:
                mix_re = Switch(sub_tree, switch=switch.outputs["zero"],
                                false=output.outputs["re"],
                                true=zero.outputs["re"],
                                input_type="FLOAT",
                                location=(left, 1.5-5*count))
                mix_im = Switch(sub_tree, switch=switch.outputs["zero"],
                                false=output.outputs["im"],
                                true=zero.outputs["im"],
                                input_type="FLOAT",
                                location=(left, 1-5*count))
                sub_tree.links.new(mix_re.outputs[0], group_outputs.inputs[0])
                sub_tree.links.new(mix_im.outputs[0], group_outputs.inputs[1])
                res.append(mix_re)
                ims.append(mix_im)



        left+=1
        # combine all Ylms to the final function
        expr=""
        first = True
        for coeff,Y in zip(coefficient_labels,Y_labels):
            if first:
                expr = coeff+","+Y+",*"
                first = False
            else:
                expr=expr+","+coeff+","+Y+",*,+"
        super_position=make_function(sub_tree,name="Superposition",
                    functions={
                        "Y":expr,
                    },inputs=coefficient_labels+Y_labels,outputs=["Y"],location=(left,0),
                    scalars=coefficient_labels+Y_labels+["Y"],node_group_type=self.node_group_type)
        for label,socket in zip(coefficient_labels,coefficient_sockets):
            sub_tree.links.new(socket,super_position.inputs[label])
            sub_tree.links.new(socket,super_position.inputs[label])
        for Y_label, Yreal, Yimag,rals in zip(Y_labels,res,ims,realpart):
            if real:
                sub_tree.links.new(Yreal.std_out,super_position.inputs[Y_label])
            else:
                sub_tree.links.new(Yimag.std_out,super_position.inputs[Y_label])

        sub_tree.links.new(super_position.outputs["Y"],group_outputs.inputs["Y"])

        self.node = group
        if isinstance(theta, (float, int)):
            self.node.inputs["theta"].default_value = theta
        else:
            tree.links.new(theta, self.node.inputs["theta"])

        if isinstance(phi, (float, int)):
            self.node.inputs["phi"].default_value = phi
        else:
            tree.links.new(phi, self.node.inputs["phi"])

        self.phi = self.node.inputs["phi"]
        self.theta = self.node.inputs["theta"]
        self.inputs=self.node.inputs
        self.outputs=self.node.outputs

class CMBNode(GenericNode):
    def __init__(self, tree, ls=[2,3,5], location=(0, 0), theta=0, phi=0, **kwargs):
        """
        This simulates the temperature distribution of the cosmic microwave background
        max_l: should be less or equal 200
        """
        max_l = max(ls)
        n= len(ls)
        self.name = get_from_kwargs(kwargs, 'name', "CMB_"+str(max_l)+"_"+str(n))
        super().__init__(tree, location=location, **kwargs)

        # create the correct node group
        if 'Geometry' in self.node_group_type:
            sub_tree = bpy.data.node_groups.new(type='GeometryNodeTree', name=self.name)
            group = tree.nodes.new(type='GeometryNodeGroup')
        else:
            sub_tree = bpy.data.node_groups.new(type='ShaderNodeTree', name=self.name)
            group = tree.nodes.new(type='ShaderNodeGroup')

        group.location = (location[0] * 200, location[1] * 100)
        group.name = self.name
        group.label = self.name
        group.node_tree = sub_tree

        # create the sockets for the group
        group_inputs = sub_tree.nodes.new('NodeGroupInput')
        group_outputs = sub_tree.nodes.new('NodeGroupOutput')
        make_new_socket(sub_tree, name="theta", io='INPUT', type='NodeSocketFloat')
        make_new_socket(sub_tree, name="phi", io='INPUT', type='NodeSocketFloat')
        make_new_socket(sub_tree, name="Y", io='OUTPUT', type='NodeSocketFloat')

        # prepare data, select random quantum numbers for l and m
        print("Spherical harmonics generated for the following multipole moments: "+str(ls))

        seed = get_from_kwargs(kwargs,'seed',int(random.random()*1000))
        print("Seed: ",seed)
        random.seed(seed)
        ms =[int(-l+2*l*random.random()) for l in ls]
        print("The following m values are selected randomly: "+str(ms))


        # create a random list for picking real or imaginary parts
        coeff_labels = []
        coeff_sockets = []
        phase = []
        for i in range(n):
            if random.random()<0.5:
                phase.append(True)
            else:
                phase.append(False)
            label = "a" + str(i)
            make_new_socket(sub_tree, name=label, io='INPUT', type='NodeSocketFloat')
            coeff_labels.append(label)
            coeff_sockets.append(group_inputs.outputs[label])

        max_m = 0
        for mabs in ms:
            if abs(mabs) > max_m:
                max_m = abs(mabs)

        group_inputs.location = (-(0.85 * max_l + 2 + max_m) * 200, 0)
        group_outputs.location = (0, 0)

        left = -(0.85 * max_l + 1 + max_m)

        xy = make_function(sub_tree.nodes,
                           functions={
                               "x": "theta,cos",
                               "y": "theta,sin"
                           },
                           name="xyTransform", location=(left, 0),
                           scalars=["x", "y", "theta"], inputs=["theta"], outputs=["x", "y"],
                           node_group_type=self.node_group_type)

        sub_tree.links.new(group_inputs.outputs["theta"], xy.inputs["theta"])
        x_socket = xy.outputs["x"]

        left = -0.85 * max_l - max_m

        left += 1
        zero = make_function(sub_tree, name="complexZero",
                                      functions={
                                          "re": "0",
                                          "im": "0"
                                      }, outputs=["re", "im"],
                                      scalars=["re", "im"], location=(left , 5),
                                      node_group_type=self.node_group_type)

        # the following nodes are independent of m and are shared between
        # associated Legendre polynomials of different m
        # make Legendre polynomials
        p0 = make_function(sub_tree.nodes,
                           functions={
                               "P0": "1"
                           },
                           location=(left, 0),
                           inputs=["x"], outputs=["P0"],
                           scalars=["x", "P0"],
                           node_group_type=self.node_group_type,
                           name="P0",
                           )
        ps = [p0]
        if max_l > 0:
            p1 = make_function(sub_tree.nodes,
                               functions={
                                   "P1": "x"
                               },
                               location=(left, 1),
                               inputs=["x"], outputs=["P1"],
                               scalars=["x", "P1"],
                               node_group_type=self.node_group_type,
                               name="P1",
                               )
            ps.append(p1)

        for l in range(2, max_l + 1):
            expr = "2," + str(l) + ",*,1,-,x,*,P" + str(l - 1) + ",*," + str(l - 1) + ",P" + str(l - 2) + ",*,-," + str(
                l) + ",/"
            p_next = make_function(sub_tree.nodes,
                                   functions={
                                       "P" + str(l): expr
                                   },
                                   location=(left + 0.85 * (l - 1), l % 2),
                                   inputs=["x", "P" + str(l - 1), "P" + str(l - 2)], outputs=["P" + str(l)],
                                   scalars=["x", "P" + str(l), "P" + str(l - 1), "P" + str(l - 2)],
                                   node_group_type=self.node_group_type,
                                   name="P" + str(l),
                                   )
            sub_tree.links.new(ps[-1].outputs[0], p_next.inputs["P" + str(l - 1)])
            sub_tree.links.new(ps[-2].outputs[0], p_next.inputs["P" + str(l - 2)])
            ps.append(p_next)

        for p in ps:
            sub_tree.links.new(x_socket, p.inputs["x"])

        sub_tree.links.new(ps[-1].outputs[0], group_outputs.inputs[0])

        # make derivatives

        left = -0.85 * max_l - max_m
        # make Legendre polynomials
        dp0 = make_function(sub_tree.nodes,
                            functions={
                                "P'0": "0"
                            },
                            location=(left, -2),
                            inputs=["x"], outputs=["P'0"],
                            scalars=["x", "P'0"],
                            node_group_type=self.node_group_type,
                            name="P'0",
                            )
        dps = [dp0]
        if max_l > 0:
            dp1 = make_function(sub_tree.nodes,
                                functions={
                                    "P'1": "1"
                                },
                                location=(left, -1),
                                inputs=["x"], outputs=["P'1"],
                                scalars=["x", "P'1"],
                                node_group_type=self.node_group_type,
                                name="P'1",
                                )
            dps.append(dp1)

        for l in range(2, max_l + 1):
            expr = "2," + str(l) + ",*,1,-,P" + str(l - 1) + ",x,P'" + str(l - 1) + ",*,+,*," + str(
                l - 1) + ",P'" + str(l - 2) + ",*,-," + str(
                l) + ",/"
            dp_next = make_function(sub_tree.nodes,
                                    functions={
                                        "P'" + str(l): expr
                                    },
                                    location=(left + 0.85 * (l - 1), -2 + l % 2),
                                    inputs=["x", "P" + str(l - 1), "P'" + str(l - 1), "P'" + str(l - 2)],
                                    outputs=["P'" + str(l)],
                                    scalars=["x", "P" + str(l - 1), "P'" + str(l), "P'" + str(l - 1),
                                             "P'" + str(l - 2)],

                                    node_group_type=self.node_group_type,
                                    name="P'" + str(l),
                                    )
            sub_tree.links.new(ps[l - 1].outputs[0], dp_next.inputs["P" + str(l - 1)])
            sub_tree.links.new(dps[-1].outputs[0], dp_next.inputs["P'" + str(l - 1)])
            sub_tree.links.new(dps[-2].outputs[0], dp_next.inputs["P'" + str(l - 2)])
            dps.append(dp_next)

        for dp in dps:
            sub_tree.links.new(x_socket, dp.inputs["x"])
        sub_tree.links.new(dps[-1].outputs[0], group_outputs.inputs[0])

        # from here on, for each spherical harmonics the node system differs
        count = 0
        fitting_functions = fitting_function_dictionary()

        res = []
        ims = []

        Y_labels = []

        for l in ls:
            # grab m value
            current_m = ms[count]
            mabs = abs(current_m)
            left = -mabs+5
            al0 = make_function(sub_tree.nodes,
                                functions={
                                    "P" + str(l) + "_0": "P" + str(l)
                                },
                                location=(left, -4-5*count),
                                inputs=["x", "y", "P" + str(l)], outputs=["P" + str(l) + "_0"],
                                scalars=["x", "y", "P" + str(l), "P" + str(l) + "_0"],
                                node_group_type=self.node_group_type,
                                name="P" + str(l) + "_0",
                                )
            sub_tree.links.new(ps[l].outputs[0], al0.inputs["P" + str(l)])


            # here comes the m-dependent part

            # this is a fitting function that returns zero in a given range of domain, where the true computation
            # would be spoiled by numerical inaccuracies
            limit = lambda x: eval(fitting_functions[l])
            lim = limit(mabs)
            print("limit for zero: ",l,current_m,lim)

            switch= make_function(sub_tree, name="zeroRangeFor"+str(current_m),
                                   functions={
                                       "zero": "theta,"+str(lim)+",<,pi,theta,-,"+str(lim)+",<,+"
                                   }, inputs=["theta"], outputs=["zero"],
                                   scalars=["theta", "zero"],
                                   location=(left, -3-5*count), node_group_type=self.node_group_type)
            sub_tree.links.new(group_inputs.outputs["theta"], switch.inputs["theta"])

            y_function = make_function(sub_tree,name="ySocketFor"+str(current_m),
                        functions={
                            "y":"1,zero,1,=,*,theta,sin,zero,0,=,*,+"
                        },inputs=["theta","zero"],outputs=["y"],location=(left+0.5,-3.5-5*count),
                        scalars=["theta","zero","y"], node_group_type=self.node_group_type)
            sub_tree.links.new(group_inputs.outputs["theta"],y_function.inputs["theta"])
            sub_tree.links.new(switch.outputs["zero"],y_function.inputs["zero"])
            y_socket = y_function.outputs["y"]

            als_m = [al0]
            # make associated Legendre functions
            # use the following recursive relation P_l^{m} = -(l+m-1)(l-m+2)P_{l}^{m-2}-2(m-1)*x/y*P_l^{m-1}
            if current_m > 0:
                al1 = make_function(sub_tree.nodes,
                                    functions={
                                        "P" + str(l) + "_1": "-1,y,*,P'" + str(l) + ",*," + str(
                                            l * (l + 1)) + ",sqrt,/"
                                    },
                                    location=(left, -5-5*count),
                                    inputs=["x", "y", "P" + str(l), "P'" + str(l)], outputs=["P" + str(l) + "_1"],
                                    scalars=["x", "y", "P" + str(l), "P'" + str(l), "P" + str(l) + "_1"],
                                    node_group_type=self.node_group_type,
                                    name="P" + str(l) + "_1")
                sub_tree.links.new(ps[l].outputs[0], al1.inputs["P" + str(l)])
                sub_tree.links.new(dps[l].outputs[0], al1.inputs["P'" + str(l)])
            else:
                # differs only by a sign from the one for positive m
                al1 = make_function(sub_tree.nodes,
                                    functions={
                                        "P" + str(l) + "_-1": "y,P'" + str(l) + ",*," + str(l * (l + 1)) + ",sqrt,/"
                                    },
                                    location=(left, -5-5*count),
                                    inputs=["x", "y", "P" + str(l), "P'" + str(l)], outputs=["P" + str(l) + "_-1"],
                                    scalars=["x", "y", "P" + str(l), "P'" + str(l), "P" + str(l) + "_-1"],
                                    node_group_type=self.node_group_type,
                                    name="P" + str(l) + "_-1")
                sub_tree.links.new(ps[l].outputs[0], al1.inputs["P" + str(l)])
                sub_tree.links.new(dps[l].outputs[0], al1.inputs["P'" + str(l)])

            als_m.append(al1)

            # use the following recursive relation P_l^{m} = -(l+m-1)(l-m+2)P_{l}^{m-2}-2(m-1)*x/y*P_l^{m-1}
            if current_m > 0:
                for m in range(2, mabs + 1):
                    expr = str((l + m - 1) * (l - m + 2)) + "," + str(
                        (l + m) * (l - m + 1)) + ",/,sqrt" + ",P" + str(l) + "_" + str(m - 2) + ",*,-1,*," + str(
                        2 * (m - 1)) + "," + str((l + m) * (l - m + 1)) + ",sqrt,/,x,*,y,/,P" + str(l) + "_" + str(
                        m - 1) + ",*,-"
                    al_next = make_function(sub_tree.nodes,
                                            functions={
                                                "P" + str(l) + "_" + str(m): expr
                                            },
                                            location=(left + 0.85 * (m - 1), -5 + m % 2-5*count),
                                            inputs=["x", "y", "P" + str(l) + "_" + str(m - 1),
                                                    "P" + str(l) + "_" + str(m - 2)],
                                            outputs=["P" + str(l) + "_" + str(m)],
                                            scalars=["x", "y", "P" + str(l) + "_" + str(m - 1),
                                                     "P" + str(l) + "_" + str(m - 2), "P" + str(l) + "_" + str(m)],

                                            node_group_type=self.node_group_type,
                                            name="P" + str(l) + "_" + str(m),
                                            )
                    sub_tree.links.new(als_m[-1].outputs[0], al_next.inputs["P" + str(l) + "_" + str(m - 1)])
                    sub_tree.links.new(als_m[-2].outputs[0], al_next.inputs["P" + str(l) + "_" + str(m - 2)])
                    als_m.append(al_next)
            else:
                for m in range(2, mabs + 1):
                    # only one sign different
                    expr = str((l + m - 1) * (l - m + 2)) + "," + str(
                        (l + m) * (l - m + 1)) + ",/,sqrt" + ",P" + str(
                        l) + "_" + str(-m + 2) + ",*,-1,*," + str(2 * (m - 1)) + "," + str(
                        (l + m) * (l - m + 1)) + ",sqrt,/,x,*,y,/,P" + str(l) + "_" + str(-m + 1) + ",*,+"
                    al_next = make_function(sub_tree.nodes,
                                            functions={
                                                "P" + str(l) + "_" + str(-m): expr
                                            },
                                            location=(left + 0.85 * (m - 1), -5 + m % 2-5*count),
                                            inputs=["x", "y", "P" + str(l) + "_" + str(-m + 1),
                                                    "P" + str(l) + "_" + str(-m + 2)],
                                            outputs=["P" + str(l) + "_" + str(-m)],
                                            scalars=["x", "y", "P" + str(l) + "_" + str(-m + 1),
                                                     "P" + str(l) + "_" + str(-m + 2),
                                                     "P" + str(l) + "_" + str(-m)],

                                            node_group_type=self.node_group_type,
                                            name="P" + str(l) + "_" + str(-m),
                                            )
                    sub_tree.links.new(als_m[-1].outputs[0], al_next.inputs["P" + str(l) + "_" + str(-m + 1)])
                    sub_tree.links.new(als_m[-2].outputs[0], al_next.inputs["P" + str(l) + "_" + str(-m + 2)])
                    als_m.append(al_next)

            for al in als_m:
                sub_tree.links.new(x_socket, al.inputs["x"])
                sub_tree.links.new(y_socket, al.inputs["y"])

            # finalizing things
            if current_m == 0:
                alp = als_m[-2]
            else:
                alp = als_m[-1]

            left += 1
            # include phi, and rescaling by
            output = make_function(sub_tree, name="Y_"+str(l)+"_"+str(current_m),
                                   functions={
                                       "re": "alp," + str(current_m) + ",phi,*,cos,*,2,/," + str(
                                           2 * l + 1) + ",pi,/,sqrt,*,-1," + str(current_m) + ",**,*",
                                       "im": "alp," + str(current_m) + ",phi,*,sin,*,2,/," + str(
                                           2 * l + 1) + ",pi,/,sqrt,*,-1," + str(current_m) + ",**,*"
                                   }, inputs=["alp", "phi"], outputs=["re", "im"],
                                   scalars=["alp", "phi", "re", "im"], location=(left, 0.5-5*count),
                                   node_group_type=self.node_group_type,hide=True)
            sub_tree.links.new(alp.outputs[0], output.inputs["alp"])
            sub_tree.links.new(group_inputs.outputs["phi"], output.inputs["phi"])
            Y_labels.append("Y"+str(l)+"_"+str(current_m))

            left += 1
            if self.node_group_type == 'Shader':
                mix_re = MixNode(sub_tree, factor=switch.outputs["zero"],
                                 caseB=zero.outputs["re"],
                                 caseA=output.outputs["re"],
                                 location=(left, 1.5-5*count),
                                 name="Real",
                                 )
                mix_im = MixNode(sub_tree, factor=switch.outputs["zero"],
                                 caseB=zero.outputs["im"],
                                 caseA=output.outputs["im"],
                                 location=(left, 1-5*count),
                                 name="Imag")
                sub_tree.links.new(mix_re.std_out, group_outputs.inputs[0])
                sub_tree.links.new(mix_im.std_out, group_outputs.inputs[1])
                res.append(mix_re)
                ims.append(mix_im)
            else:
                mix_re = Switch(sub_tree, switch=switch.outputs["zero"],
                                false=output.outputs["re"],
                                true=zero.outputs["re"],
                                input_type="FLOAT",
                                location=(left, 1.5-5*count),name="Real")
                mix_im = Switch(sub_tree, switch=switch.outputs["zero"],
                                false=output.outputs["im"],
                                true=zero.outputs["im"],
                                input_type="FLOAT",
                                location=(left, 1-5*count),name="Imag")
                sub_tree.links.new(mix_re.outputs[0], group_outputs.inputs[0])
                sub_tree.links.new(mix_im.outputs[0], group_outputs.inputs[1])
                res.append(mix_re)
                ims.append(mix_im)

            count+=1



        left+=1
        # combine all Ylms to the final function
        expr=""
        first = True
        for coeff,Y in zip(coeff_labels,Y_labels):
            if first:
                expr = coeff+","+Y+",*"
                first = False
            else:
                expr=expr+","+coeff+","+Y+",*,+"
        super_position=make_function(sub_tree,name="Superposition",
                    functions={
                        "Y":expr,
                    },inputs=coeff_labels+Y_labels,outputs=["Y"],location=(left,0),
                    scalars=coeff_labels+Y_labels+["Y"],node_group_type=self.node_group_type)
        for label,socket in zip(coeff_labels,coeff_sockets):
            sub_tree.links.new(socket,super_position.inputs[label])
            sub_tree.links.new(socket,super_position.inputs[label])

        for Y_label, Yreal, Yimag, real in zip(Y_labels,res,ims,phase):
            if real:
                sub_tree.links.new(Yreal.std_out,super_position.inputs[Y_label])
            else:
                sub_tree.links.new(Yimag.std_out,super_position.inputs[Y_label])

        sub_tree.links.new(super_position.outputs["Y"],group_outputs.inputs["Y"])

        self.node = group
        if isinstance(theta, (float, int)):
            self.node.inputs["theta"].default_value = theta
        else:
            tree.links.new(theta, self.node.inputs["theta"])

        if isinstance(phi, (float, int)):
            self.node.inputs["phi"].default_value = phi
        else:
            tree.links.new(phi, self.node.inputs["phi"])

        self.phi = self.node.inputs["phi"]
        self.theta = self.node.inputs["theta"]
        self.inputs=self.node.inputs
        self.outputs=self.node.outputs