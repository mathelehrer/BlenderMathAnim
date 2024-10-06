import bpy

from geometry_nodes.nodes import make_function
from interface.ibpy import make_new_socket
from utils.kwargs import get_from_kwargs


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
    def __init__(self,tree,l=3,m=2,location=(0,0),x=0,**kwargs):
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

        group_inputs.location=(-(0.5*l+1)*200,0)
        group_outputs.location=(0,0)

        left = -0.5*l
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

        # make derivatives

        left = -0.5 * l
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
                                   location=(left + 0.5 * (l - 1),-2+ l % 2),
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

        # make associated Legendre functions
        # use the following recursive relation y P^(m+1)_l=(l-m)xP^m_l-(l+m)P^m_{l-1}
        al0 = make_function(sub_tree.nodes,
            functions={
                "aP0":"1"
            },
            location =(left,0),
            inputs=["x"],outputs=["P0"],
            scalars=["x","P0"],
            node_group_type=self.node_group_type,
            name="P0",
        )
        ps = [p0]


        self.node = group

        if isinstance(x, (float, int)):
            self.node.inputs["x"].default_value = x
        else:
            tree.links.new(x, self.node.inputs["x"])

        self.std_out = self.node.outputs[0]