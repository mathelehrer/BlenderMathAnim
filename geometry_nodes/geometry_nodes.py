### This is a library file, where everything connected geometry notes should go in
### The content is very project-dependent, should be excluded from the full library at some point
### only for demonstation purposes


from functools import partial

import bpy
import numpy as np

from geometry_nodes.nodes import MeshLine, Grid, InstanceOnPoints, IcoSphere, SetShadeSmooth, Position, \
    add_locations, InputValue, SetMaterial, RealizeInstances, JoinGeometry, Points, InputBoolean, InputVector, \
    PointsToVertices, ExtrudeMesh, SetPosition, WireFrame, create_geometry_line, Index, StoredNamedAttribute, \
    ConvexHull, \
    RayCast, BooleanMath, InsideConvexHull, DeleteGeometry, NamedAttribute, VectorMath, ScaleElements, make_function, \
    Rotation, Transpose, LinearMap, TransformGeometry, layout, CubeMesh, InsideConvexHull3D, CombineXYZ, E8Node, Matrix, \
    ProjectionMap, DomainSize, SampleIndex, RepeatZone, MergeByDistance, MeshToPoints, Simulation
from interface.ibpy import Vector, create_group_from_vector_function, if_node, get_color_from_string, make_new_socket, \
    get_material, create_group_from_scalar_function
from appearance.textures import penrose_material, create_material_for_e8_visuals, star_color, decay_mode_material, \
    z_gradient
from mathematics.groups.e8 import E8Lattice
from mathematics.mathematica.mathematica import choose, tuples
from physics.constants import decay_modes
from utils.utils import flatten


############################
### auxiliary functions ####
############################

def get_parameter(node_group, node_name, input=None, output=None):
    node = None
    if node_name in node_group.nodes:
        node = node_group.nodes.get(node_name)
    else:
        for n in node_group.nodes:
            if node_name in n.label:
                node = n
    if node:
        if input is not None:
            return node.inputs[input]
        if output is not None:
            return node.outputs[output]


def setup_geometry_nodes(name,group_input=False):
    """
    auxiliary function to provide boilerplate code
    should be used more frequently,
    haven't got accustomed to it yet
    it's a little awkward, the group_output_node has to collected from the tree
    for the creation of the geometry line
    out = node_tree.nodes.get("Group Output"), when this function is used
    """
    node_tree = bpy.data.node_groups.new(name, type='GeometryNodeTree')
    nodes = node_tree.nodes

    nodes.new('NodeGroupOutput')
    make_new_socket(node_tree, name='Geometry', io='OUTPUT', type='NodeSocketGeometry')

    if group_input:
        nodes.new('NodeGroupInput')
        make_new_socket(node_tree, name='Geometry', io='INPUT', type='NodeSocketGeometry')

    return node_tree


#######################
### Example nodes #####
#######################

###
## The following node constructions are usually restricted to one particular purpose
## The reusability of these functions is highly unlikely
## Maybe they should rather be collected in the project folders
###

######################
## Demonstration of geometry nodes for displaying functions
######################
def create_node_for_optimization():
    node_tree = bpy.data.node_groups.new("SurfaceOfTheFunction", type='GeometryNodeTree')
    tree = node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    left = -8
    # boiler-plate code, create input and output sockets
    group_outputs = nodes.new('NodeGroupOutput')
    make_new_socket(node_tree, name='Geometry', io='OUTPUT', type='NodeSocketGeometry')

    # create grid, the (x,y) domain
    grid = Grid(tree, location=(left, 1), size_x=10, size_y=10, vertices_x=1000, vertices_y=1000)
    # instead of using 1000 vertices in each direction, one could also use a subdivision node

    # read in the original location of each vertex (the x,y coordinates are kept and the z coordinate is adjusted
    # depending on the value of the function f(x,y)
    pos = Position(tree, location=(left, 0))

    left += 1

    # create the function, here is your function $\left(2(x-3)^2+(y-2)^2-1\right) \cdot(x-1)(y-1)+3$
    surface = make_function(tree, functions={
        "position": [
            "pos_x",  # x stays unchanged
            "pos_y",  # y stays unchanged
            "2,pos_x,3,-,2,**,*,pos_y,2,-,2,**,+,1,-,pos_x,1,-,pos_y,1,-,*,*,3,+"  # f(x,y)
        ]
    }, name="Surface", location=(left, 0), inputs=["pos"], outputs=["position"], vectors=["pos", "position"])

    links.new(pos.std_out, surface.inputs["pos"])
    left += 1

    # window: -5<=f(x,y)<=5
    selector = make_function(tree, functions={
        "deselect": "pos_z,-5,>,pos_z,5,<,and,not"
    }, name="Window", location=(left, 1), inputs=["pos"], outputs=["deselect"], vectors=["pos"], scalars=["deselect"])
    links.new(surface.outputs["position"], selector.inputs["pos"])

    left += 1
    del_geo = DeleteGeometry(tree, location=(left, 0), selection=selector.outputs["deselect"])

    left += 1
    # update the z-value for each mesh point
    set_pos = SetPosition(tree, location=(left, 0), position=surface.outputs["position"])

    left += 1
    smooth = SetShadeSmooth(tree, location=(left, 0))

    left += 1

    # custom material that is a gradient in z-direction
    mat = SetMaterial(tree, location=(left, 0), material=z_gradient, roughness=0.005, metallic=0.45, emission=0.1)
    # connect all geometry nodes
    create_geometry_line(tree, [grid, del_geo, set_pos, smooth, mat], out=group_outputs.inputs["Geometry"])

    return node_tree


######################
## Nodes for the CMB project
#######################
def create_spectral_class(type, r=10000):
    """
    Create a sky-dome, that contains all stars of the catalog with the given spectral class

    :param r: radius of the sky-dome
    :param type: spectral class
    :param val: declination, right ascension, magnitude of the stars
    :return:
    """

    node_tree = bpy.data.node_groups.new("SpectralClass" + str(type), type='GeometryNodeTree')
    tree = node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    left = -8

    group_outputs = nodes.new('NodeGroupOutput')
    make_new_socket(node_tree, name='Geometry', io='OUTPUT', type='NodeSocketGeometry')

    group_inputs = nodes.new('NodeGroupInput')
    group_inputs.location = (left * 200, 0)
    left += 1
    make_new_socket(node_tree, name='Geometry', io='INPUT', type='NodeSocketGeometry')

    pos = Position(tree, location=(left, 0))
    unit_x = InputVector(tree, location=(left, -0.5), value=[1, 0, 0])
    angle_projection = InputVector(tree, location=(left, -1), value=[0, -1,
                                                                     1])  # this vector projects out the two relevant components for declination and right ascension
    radius = InputValue(tree, location=(left, 0.5), value=r)
    sphere = IcoSphere(tree, location=(left, 2), radius=50, subdivisions=2)

    left += 1
    # pos_x is the magnitude of the star
    # it is raised to some power 1.35 to damp faint stars
    conversion = make_function(tree, functions={
        "Position": "unit_x,pos,angle_projection,mul,rot,r,scale",
        "Scale": "1,pos_x,1.15,**,1,max,/"
    }, name="Conversion", hide=True, location=(left, -1),
                               inputs=["r", "pos", "unit_x", "angle_projection"], outputs=["Position", "Scale"],
                               scalars=["r", "Scale"], vectors=["pos", "unit_x", "Position", "angle_projection"]
                               )

    for node, label in zip([radius, pos, unit_x, angle_projection], ["r", "pos", "unit_x", "angle_projection"]):
        links.new(node.std_out, conversion.inputs[label])

    left += 1

    iop = InstanceOnPoints(tree, location=(left, 0),
                           instance=sphere.geometry_out,
                           scale=conversion.outputs["Scale"],
                           points=group_inputs.outputs['Geometry'])

    left += 1
    set_pos = SetPosition(tree, location=(left, 0),
                          position=conversion.outputs['Position'])

    left += 1
    material = partial(star_color, type=type, emission=1)
    set_mat = SetMaterial(tree, material=material, location=(left, 0))
    left += 1
    smooth = SetShadeSmooth(tree, location=(left, 0))
    create_geometry_line(tree, [iop, set_pos, set_mat, smooth], out=group_outputs.inputs['Geometry'])

    return node_tree

def create_pendulum_node():
    tree= setup_geometry_nodes("Pendulum",group_input=True)
    links = tree.links
    left = -16

    origin = Points(tree,location=(left,4),name="Origin")
    point = Points(tree,location=(left,0),name="PendulumMass")
    left+=1

    simulation=Simulation(tree,location=(left,0))
    simulation.add_socket(socket_type='FLOAT',name="theta") # elongation
    simulation.add_socket(socket_type='FLOAT',name="omega") # angular velocity
    out = tree.nodes.get("Group Output")
    ins = tree.nodes.get("Group Input")
    make_new_socket(tree,name="theta",io="INPUT")
    make_new_socket(tree, name="omega", io="INPUT")
    make_new_socket(tree, name="b", io="INPUT")# damping to compensate accumulating errors
    ins.location=((left-1)*200,-200)
    links.new(ins.outputs["omega"],simulation.simulation_input.inputs["omega"])
    links.new(ins.outputs["theta"],simulation.simulation_input.inputs["theta"])

    length = InputValue(tree, location=(left - 1, -4), value=2.5)

    left+=2
    update_omega=make_function(tree,functions={
        "omega":"o,9.81,l,/,theta,sin,*,b,o,*,+,dt,*,-"
    },name="updateOmega",location=(left,-0.5),hide=True,
    outputs=["omega"],inputs=["dt","theta","o","l","b"],scalars=["omega","o","l","theta","dt","b"])

    links.new(length.std_out,update_omega.inputs["l"])
    links.new(ins.outputs["b"],update_omega.inputs["b"])
    links.new(simulation.simulation_input.outputs["theta"],update_omega.inputs["theta"])
    links.new(simulation.simulation_input.outputs["Delta Time"],update_omega.inputs["dt"])
    links.new(update_omega.outputs["omega"],simulation.simulation_output.inputs["omega"])
    links.new(simulation.simulation_input.outputs["omega"],update_omega.inputs["o"])

    update_theta = make_function(tree, functions={
        "theta": "th,omega,dt,*,+"
    }, name="updateTheta", location=(left, -1.5),hide=True,
                                 outputs=["theta"], inputs=["dt", "th", "omega"],
                                 scalars=["theta", "th", "dt", "omega"])

    links.new(simulation.simulation_input.outputs["theta"], update_theta.inputs["th"])
    links.new(simulation.simulation_input.outputs["Delta Time"], update_theta.inputs["dt"])
    links.new(update_theta.outputs["theta"], simulation.simulation_output.inputs["theta"])
    links.new(simulation.simulation_input.outputs["omega"], update_theta.inputs["omega"])
    left+=4


    # convert angle into position
    converter = make_function(tree,functions={
        "position":["theta,sin,l,*","0","theta,cos,l,-1,*,*"]
    },name="Angle2Position",inputs=["theta","l"],outputs=["position"],scalars=["l","theta"],vectors=["position"],
                              location=(left,-1))
    links.new(simulation.simulation_output.outputs["theta"],converter.inputs["theta"])
    links.new(length.std_out,converter.inputs["l"])
    left+=1

    set_pos = SetPosition(tree,position=converter.outputs["position"],location=(left,-0.5))
    left+=1
    join = JoinGeometry(tree,location=(left,0))
    create_geometry_line(tree,[origin,join])

    left+=1
    point2mesh = PointsToVertices(tree,location=(left,0))
    left+=1
    convex_hull =ConvexHull(tree,location=(left,0))
    left += 1
    wireframe = WireFrame(tree, location=(left, 0))
    left+=1
    mat = SetMaterial(tree,location=(left,0),material='joker')
    left+=1
    join2 = JoinGeometry(tree,location=(left,0))
    left+=1
    trafo = TransformGeometry(tree,location=(left,0),translation=[5,0,1])
    left+=1
    create_geometry_line(tree,[point,simulation,set_pos,join,point2mesh,
                               convex_hull,wireframe,mat,join2,trafo],out=out.inputs["Geometry"])

    # create branch for the mass
    left-=6
    pos = Position(tree,location =(left,-1) )
    left+=1
    length = VectorMath(tree,location=(left,-1),operation='LENGTH',inputs0=pos.std_out)# one vertex is at the origin
    # only the vertex away from the origin has a non-zero length, this value is used to select the correct vertex for the instance on points
    uv = IcoSphere(tree,location=(left,-2),radius=0.3,subdivisions=3)
    left+=1
    iop = InstanceOnPoints(tree,location=(left,-1),selection=length.std_out,instance=uv.geometry_out)
    left+=1
    mat2 = SetMaterial(tree,location=(left,-2),material='plastic_example')
    create_geometry_line(tree,[point2mesh,iop,mat2,join2])
    return tree

#########################
## Nuclide table (chemistry with ChemNerd44) pending
##########################

def create_nuclid_nodes(size=1):
    """
    Create a 3D visualization of the nuclid table
    :return:
    """

    node_tree = bpy.data.node_groups.new("NuclidTable", type='GeometryNodeTree')
    tree = node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    left = -8

    group_outputs = nodes.new('NodeGroupOutput')
    make_new_socket(node_tree, name='Geometry', io='OUTPUT', type='NodeSocketGeometry')

    group_inputs = nodes.new('NodeGroupInput')
    group_inputs.location = (left * 200, 0)
    left += 1
    make_new_socket(node_tree, name='Geometry', io='INPUT', type='NodeSocketGeometry')

    # without this silly conversion the blender file crashes
    mesh2points = MeshToPoints(tree, location=(left, 0))
    points2mesh = PointsToVertices(tree, location=(left + 1, 0))

    left += 2

    extrude = ExtrudeMesh(tree, mode='VERTICES', location=(left, 0), offset=Vector([0.95 * size, 0, 0]))
    height_attribute = NamedAttribute(tree, location=(left, -1), name='HalfLife')
    left += 1
    extrude2 = ExtrudeMesh(tree, mode='EDGES', location=(left, 0), offset=Vector([0, 0.95 * size, 0]))
    combine = CombineXYZ(tree, location=(left, -1))
    tree.links.new(height_attribute.std_out, combine.std_in[2])
    left += 1
    extrude3 = ExtrudeMesh(tree, mode='FACES', location=(left, 0))
    tree.links.new(combine.std_out, extrude3.inputs["Offset"])
    left += 1
    set_mat = SetMaterial(tree, location=(left, 0),
                          material=partial(decay_mode_material, roughness=0.25, metallic=0.4, emission=1))
    create_geometry_line(tree, [mesh2points, points2mesh, extrude, extrude2, extrude3, set_mat],
                         ins=group_inputs.outputs["Geometry"], out=group_outputs.inputs['Geometry'])

    return node_tree


def create_flat_nuclid_nodes(size=1):
    """
    Create a 2D visualization of the nuclid table
    :return:
    """

    node_tree = bpy.data.node_groups.new("NuclidTable", type='GeometryNodeTree')
    tree = node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    left = -10

    group_outputs = nodes.new('NodeGroupOutput')
    make_new_socket(node_tree, name='Geometry', io='OUTPUT', type='NodeSocketGeometry')

    group_inputs = nodes.new('NodeGroupInput')
    group_inputs.location = (left * 200, 0)
    left += 1
    make_new_socket(node_tree, name='Geometry', io='INPUT', type='NodeSocketGeometry')

    z_min = InputValue(tree, name="ZMin", value=0, location=(left, 0.5))
    z_max = InputValue(tree, name="ZMax", value=0.9999 * size, location=(left, 0.))

    stable_selector = InputBoolean(tree, name="Stable", value=True, location=(left + 1, 0.25))
    growers = []
    grower_labels = []
    for i, mode in enumerate(decay_modes):
        if i > 0:  # skip stable mode
            growers.append(InputValue(tree, name=mode + "Grower", value=40, location=(left, -i * 0.5)))
            grower_labels.append(mode + "Grower")

    attr_mode = NamedAttribute(tree, location=(left, -5), name='DecayMode')
    attr_time = NamedAttribute(tree, location=(left, -5.5), name='HalfLife')
    position = Position(tree, location=(left, -6))

    left += 2

    modi = list(decay_modes.keys())
    term = ""
    for i, mode in enumerate(modi):
        if i > 0:
            part = "attr_mode," + str(i) + ",=,attr_time,lg," + grower_labels[i - 1] + ",>,and"
        else:  # the display of the stable isotopes is only controlled by z_min and z_max
            part = "attr_mode," + str(0) + ",=," + mode + ",and"
        if decay_modes[mode] > 0:
            term = term + "," + part + ",or"
        else:
            term = part

    select_function = make_function(tree, functions={
        "Selection": term + ",pos_y,z_min,>,and,pos_y,z_max,<,and"
    },
                                    name="SelectionFunction", location=(left, -1),
                                    inputs=["stable", "z_min", "z_max", "pos", "attr_mode",
                                            "attr_time"] + grower_labels,
                                    scalars=["stable", "z_min", "z_max", "Selection", "attr_mode",
                                             "attr_time"] + grower_labels, vectors=["pos"], outputs=["Selection"])

    links.new(attr_mode.std_out, select_function.inputs["attr_mode"])
    links.new(attr_time.std_out, select_function.inputs["attr_time"])

    links.new(stable_selector.std_out, select_function.inputs["stable"])
    links.new(position.std_out, select_function.inputs["pos"])
    for grower, label in zip(growers, grower_labels):
        if grower:
            links.new(grower.std_out, select_function.inputs[grower.name])
    links.new(z_min.std_out, select_function.inputs["z_min"])
    links.new(z_max.std_out, select_function.inputs["z_max"])

    # without this silly conversion the blender file crashes
    mesh2points = MeshToPoints(tree, location=(left, 0), selection=select_function.outputs["Selection"])
    points2mesh = PointsToVertices(tree, location=(left + 1, 0))

    left += 2

    extrude = ExtrudeMesh(tree, mode='VERTICES', location=(left, 0), offset=Vector([0.95 * size, 0, 0]))

    left += 1
    extrude2 = ExtrudeMesh(tree, mode='EDGES', location=(left, 0), offset=Vector([0, 0.95 * size, 0]))

    left += 1
    extrude3 = ExtrudeMesh(tree, mode='FACES', location=(left, 0), offset=[0, 0, 1])
    left += 1
    set_mat = SetMaterial(tree, location=(left, 0), material=partial(decay_mode_material, roughness=0.25, metallic=0.4))
    create_geometry_line(tree, [mesh2points, points2mesh, extrude, extrude2,
                                extrude3,
                                set_mat], ins=group_inputs.outputs["Geometry"], out=group_outputs.inputs['Geometry'])

    return node_tree


##########################
## E8 geometry (only a short so far)
###########################

def create_e8():
    """
       create a geometry node that generates the e8 lattice
       :return:
       """
    node_tree = bpy.data.node_groups.new("E8LatticeNode", type='GeometryNodeTree')

    nodes = node_tree.nodes
    links = node_tree.links

    left = -20

    group_outputs = nodes.new('NodeGroupOutput')
    make_new_socket(node_tree, name='Geometry', io='OUTPUT', type='NodeSocketGeometry')

    # create E8 vertices (hard coded)
    e8_node = E8Node(node_tree, location=(left + 2, 0))
    e8 = E8Lattice()

    for i in range(4, 0, -1):
        theta = InputValue(node_tree, name='theta_' + str(i), location=(left - 2 * i - 1, -10))
        print("Create rotation " + str(i) + "... ", end='')
        rotation = Rotation(node_tree, dimension=8, angle=0, u=2 * (i - 1), v=2 * i - 1, location=(left - 2 * i, 4),
                            hide=False, name="Rotation" + str(i))
        node_tree.links.new(theta.std_out, rotation.inputs['Angle'])
        print("done")
        print("Create linear maps u and v " + str(i) + " ... ", end='')
        linear_map_u = LinearMap(node_tree, dimension=8, location=(left - 2 * i, 6), name="MapU" + str(i))
        linear_map_v = LinearMap(node_tree, dimension=8, location=(left - 2 * i, 3), name="MapV" + str(i))
        print("done")
        if i == 4:
            entries = e8.coxeter_plane()
            linear_map_u.inputs["v_0"].default_value = entries[0][0:3]
            linear_map_u.inputs["v_1"].default_value = entries[0][3:6]
            linear_map_u.inputs["v_2"].default_value = entries[0][6:8] + [0]

            linear_map_v.inputs["v_0"].default_value = entries[1][0:3]
            linear_map_v.inputs["v_1"].default_value = entries[1][3:6]
            linear_map_v.inputs["v_2"].default_value = entries[1][6:8] + [0]
        else:
            for j in range(3):
                node_tree.links.new(old_v.outputs["v_" + str(j)], linear_map_v.inputs["v_" + str(j)])
                node_tree.links.new(old_u.outputs["v_" + str(j)], linear_map_u.inputs["v_" + str(j)])
        old_u = linear_map_u
        old_v = linear_map_v

        for j in range(24):
            node_tree.links.new(rotation.outputs[j], linear_map_u.inputs[j + 1])
            node_tree.links.new(rotation.outputs[j], linear_map_v.inputs[j + 1])

    projectionMap = ProjectionMap(node_tree, location=(left + 2, -2), in_dimension=8, out_dimension=2,
                                  name='ProjectionMap')
    namedAttribute1 = NamedAttribute(node_tree, location=(left, -1), data_type='FLOAT_VECTOR', name='comp123')
    namedAttribute2 = NamedAttribute(node_tree, location=(left, -2), data_type='FLOAT_VECTOR', name='comp456')
    namedAttribute3 = NamedAttribute(node_tree, location=(left, -3), data_type='FLOAT_VECTOR', name='comp78')

    links.new(namedAttribute1.std_out, projectionMap.inputs["vi_0"])
    links.new(namedAttribute2.std_out, projectionMap.inputs["vi_1"])
    links.new(namedAttribute3.std_out, projectionMap.inputs["vi_2"])

    for i in range(3):
        links.new(linear_map_u.outputs[i], projectionMap.inputs[i])
        links.new(linear_map_v.outputs[i], projectionMap.inputs[i + 3])

    left += 4

    offset = make_function(node_tree, location=(left - 1, -3),
                           functions={
                               "offset": ["0", "0", "1,p,length,/,1,min"]
                           }, hide=True, vectors=["offset", "p"], inputs=["p"], outputs=["offset"], name="Offset")
    links.new(projectionMap.outputs["vo_0"], offset.inputs["p"])
    set_pos = SetPosition(node_tree, location=(left, -2), position=projectionMap.outputs[0],
                          offset=offset.outputs["offset"])
    points2verts = PointsToVertices(node_tree, location=(left + 1, -2))
    scale_geo = TransformGeometry(node_tree, location=(left + 2, -2), scale=[5, 5, 0.25])

    left += 3

    idx = Index(node_tree, location=(left, 0))
    domain_size = DomainSize(node_tree, location=(left, -1))
    create_geometry_line(node_tree, [points2verts, domain_size])
    pos = Position(node_tree, location=(left, -3))
    left += 1

    # retrieve eight-dimensional coordinates for edge selection
    e8coords = []
    for count, attr in enumerate(["comp123", "comp456", "comp78"]):
        namedAttr = NamedAttribute(node_tree, location=(left, 4 - count), name=attr)
        sampleIndex = SampleIndex(node_tree, location=(left + 1, 4 - count), value=namedAttr.std_out, index=idx.std_out)
        create_geometry_line(node_tree, [e8_node, sampleIndex])
        e8coords.append(sampleIndex)

    # pairing vertices with repeat zone
    repeat = RepeatZone(node_tree, location=(left, 0), width=5)
    repeat.add_socket(socket_type="INT", name='i')
    links.new(domain_size.outputs["Point Count"], repeat.inputs["Iterations"])
    addOne = make_function(node_tree, functions={
        "out": "in,1,+"
    }, scalars=["in", "out"], inputs=["in"], outputs=["out"], hide=True, location=(left + 1, -0.25), name="i++")
    links.new(repeat.repeat_input.outputs["i"], addOne.inputs["in"])
    links.new(addOne.outputs["out"], repeat.repeat_output.inputs["i"])
    e8coords2 = []
    for count, attr in enumerate(["comp123", "comp456", "comp78"]):
        namedAttr = NamedAttribute(node_tree, location=(left, -4 - count), name=attr)
        sampleIndex = SampleIndex(node_tree, location=(left + 1, -4 - count), value=namedAttr.std_out,
                                  index=repeat.outputs["i"])
        create_geometry_line(node_tree, [e8_node, sampleIndex])
        e8coords2.append(sampleIndex)

    # edge selection process
    edge_selection = make_function(node_tree, name="EdgeSelection", location=(left + 2, -2), functions={
        "select": "v00,v10,sub,v00,v10,sub,dot,v01,v11,sub,v01,v11,sub,dot,add,v02,v12,sub,v01,v12,sub,dot,add,2.01,<,i,idx,>,and"
    },
                                   inputs=["v00", "v01", "v02", "v10", "v11", "v12", "i", "idx"], outputs=["select"],
                                   vectors=["v00", "v01", "v02", "v10", "v11", "v12"],
                                   scalars=["select", "i", "idx"], hide=True)

    links.new(e8coords[0].std_out, edge_selection.inputs["v00"])
    links.new(e8coords[1].std_out, edge_selection.inputs["v01"])
    links.new(e8coords[2].std_out, edge_selection.inputs["v02"])
    links.new(e8coords2[0].std_out, edge_selection.inputs["v10"])
    links.new(e8coords2[1].std_out, edge_selection.inputs["v11"])
    links.new(e8coords2[2].std_out, edge_selection.inputs["v12"])
    links.new(repeat.repeat_input.outputs["i"], edge_selection.inputs["i"])
    links.new(idx.std_out, edge_selection.inputs["idx"])

    # extrude selected edges
    sample_idx = SampleIndex(node_tree, location=(left + 2, -3), index=repeat.repeat_input.outputs["i"],
                             value=pos.std_out, geometry=scale_geo.geometry_out)
    extrude = ExtrudeMesh(node_tree, location=(left + 3, -2), selection=edge_selection.outputs['select'])
    set_pos2 = SetPosition(node_tree, location=(left + 4, -2), selection=extrude.outputs['Top'],
                           position=sample_idx.std_out)
    # create repeat-zone interal geometry line
    repeat.create_geometry_line([extrude, set_pos2])

    left += 6
    # after configurations
    merge = MergeByDistance(node_tree, location=(left, 0))
    # edges
    wireframe = WireFrame(node_tree, location=(left + 1, 1), radius=0.0005)
    mat = SetMaterial(node_tree, material='plastic_joker', location=(left + 2, 1), alpha=0.5)
    # vertices pipe them in from scale_geo directly, to avoid averaged attributed from the MergeByDistance node
    sphere = IcoSphere(node_tree, location=(left + 0.5, -2), subdivisions=2, radius=0.075)
    iop = InstanceOnPoints(node_tree, location=(left + 1, -1), instance=sphere.geometry_out)
    mat2 = SetMaterial(node_tree, location=(left + 2, -1), material=create_material_for_e8_visuals,
                       attribute_names=["comp123", "comp456", "comp78"])
    left += 3
    join = JoinGeometry(node_tree, location=(left, 0))
    left += 1
    group_outputs.location = (left * 200, 0)
    create_geometry_line(node_tree, [e8_node, set_pos, points2verts, scale_geo, repeat])
    create_geometry_line(node_tree, [repeat, merge, wireframe, mat, join], out=group_outputs.inputs['Geometry'])
    create_geometry_line(node_tree, [scale_geo, iop, mat2, join], out=group_outputs.inputs['Geometry'])

    return node_tree


##########################
## Mandelbrot area
###########################

def polygon_group(radius=1, n_gon=6, name="PolygonGroup", colors=['text', 'gray_2']):
    """
    create a group that interpolates a disc between 6n-gons

    :param material:
    :param name:
    :return:
    """
    node_tree = bpy.data.node_groups.new(name, type='GeometryNodeTree')

    nodes = node_tree.nodes
    links = node_tree.links

    left = -1000
    width = 200

    group_inputs = nodes.new('NodeGroupInput')
    group_inputs.location = (left, 0)

    group_outputs = nodes.new('NodeGroupOutput')
    group_outputs.location = (-left, 0)

    node_tree.inputs.new('NodeSocketGeometry', 'In')
    node_tree.outputs.new('NodeSocketGeometry', 'Out')
    node_tree.outputs.new('NodeSocketColor', 'FaceColorInformation')

    # input data nodes
    pos = nodes.new(type="GeometryNodeInputPosition")
    pos.location = (left + width, -100)
    pos.hide = True

    nGon = nodes.new(type="ShaderNodeValue")
    nGon.name = 'nGon'
    nGon.label = 'nGon'
    nGon.outputs[0].default_value = n_gon
    nGon.location = (left + width, -150)
    nGon.hide = True

    index = nodes.new(type="GeometryNodeInputIndex")
    index.location = (left + width, -200)
    index.hide = True

    nGon.name = 'nGon'
    nGon.label = 'nGon'
    nGon.outputs[0].default_value = n_gon
    nGon.location = (left + width, 0)

    set_pos = nodes.new(type='GeometryNodeSetPosition')
    set_pos.location = (left + 7 * width, 100)

    links.new(group_inputs.outputs[0], set_pos.inputs['Geometry'])
    links.new(set_pos.outputs['Geometry'], group_outputs.inputs[0])

    phi = create_group_from_vector_function(nodes, functions=['y,x,atan2'], name='phi', node_group_type='Geometry')
    phi.location = (left + 2 * width, -200)
    links.new(pos.outputs[0], phi.inputs['In'])

    z0 = create_group_from_vector_function(nodes, functions=['n,phi,*,2,pi,*,/,floor,2,pi,*,n,/,*,cos,radius,*',
                                                             'n,phi,*,2,pi,*,/,floor,2,pi,*,n,/,*,sin,radius,*'],
                                           scalar_parameters=['radius', 'phi', 'n'], node_group_type='Geometry',
                                           name='z0')
    z0.location = (left + 3 * width, -200)
    z0.inputs['radius'].default_value = radius
    z0.hide = True
    links.new(nGon.outputs['Value'], z0.inputs['n'])

    links.new(phi.outputs[0], z0.inputs['phi'])

    z1 = create_group_from_vector_function(nodes, functions=['n,phi,*,2,pi,*,/,ceil,2,pi,*,n,/,*,cos,radius,*',
                                                             'n,phi,*,2,pi,*,/,ceil,2,pi,*,n,/,*,sin,radius,*'],
                                           scalar_parameters=['radius', 'phi', 'n'], node_group_type='Geometry',
                                           name='z1')
    z1.location = (left + 3 * width, -300)
    z1.inputs['radius'].default_value = radius
    z1.hide = True
    links.new(nGon.outputs['Value'], z1.inputs['n'])
    links.new(phi.outputs[0], z1.inputs['phi'])

    r = create_group_from_vector_function(nodes, functions=[
        'z1_x,z0_y,*,z0_x,z1_y,*,-,z0_y,z1_y,-,phi,cos,*,z1_x,z0_x,-,phi,sin,*,+,/,v,length,min'],
                                          parameters=['z0', 'z1'], scalar_parameters=['phi'], name='r',
                                          node_group_type='Geometry')
    r.location = (left + 4 * width, -250)
    r.hide = True
    links.new(pos.outputs[0], r.inputs['In'])
    links.new(phi.outputs[0], r.inputs['phi'])
    links.new(z1.outputs[0], r.inputs['z1'])
    links.new(z0.outputs[0], r.inputs['z0'])

    polar2cartesian = create_group_from_vector_function(nodes, functions=['r,phi,cos,*', 'r,phi,sin,*'],
                                                        scalar_parameters=['r', 'phi'], name='polar2cartesian',
                                                        node_group_type='Geometry')
    polar2cartesian.location = (left + 5 * width, -250)
    polar2cartesian.hide = True
    links.new(phi.outputs[0], polar2cartesian.inputs['phi'])
    links.new(r.outputs[0], polar2cartesian.inputs['r'])

    ifnode = if_node(nodes,
                     bool_function='z0,z1,sub,length,0.1,>,pos,length,0.1,<,+',
                     parameters=['pos', 'z0', 'z1', 'True', 'False'],
                     node_type='Geometry')
    ifnode.location = (left + 6 * width, -300)
    ifnode.hide = True
    links.new(pos.outputs[0], ifnode.inputs['pos'])
    links.new(z0.outputs[0], ifnode.inputs['z0'])
    links.new(z1.outputs[0], ifnode.inputs['z1'])
    links.new(z0.outputs[0], ifnode.inputs['False'])
    links.new(polar2cartesian.outputs[0], ifnode.inputs['True'])
    links.new(ifnode.outputs[0], set_pos.inputs['Position'])

    # checker color faces

    checker = create_group_from_vector_function(nodes, functions=['idx,n,*,1002,/,1,%,round'],
                                                scalar_parameters=['idx', 'n'],
                                                name='checkerFunction', node_group_type='Geometry')
    checker.location = (left + 3 * width, 300)
    links.new(nGon.outputs[0], checker.inputs['n'])
    links.new(index.outputs[0], checker.inputs['idx'])

    ramp = nodes.new(type='ShaderNodeValToRGB')
    ramp.location = (left + 5 * width, 200)
    ramp.color_ramp.elements[0].color = get_color_from_string(colors[0])
    ramp.color_ramp.elements[1].color = get_color_from_string(colors[1])
    links.new(checker.outputs[0], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], group_outputs.inputs['FaceColorInformation'])
    node_tree.outputs['FaceColorInformation'].attribute_domain = 'FACE'

    return node_tree


def general_monte_carlo_node(material='text',
                             resolution=10, minimum=-1, maximum=1, name="MonteCarloGroup"):
    """
    generates random spheres over the full range of the corresponding object

    :param resolution:
    :param material:
    :param maximum:
    :param minimum:
    :param name:
    """

    node_tree = bpy.data.node_groups.new(name, type='GeometryNodeTree')

    nodes = node_tree.nodes
    links = node_tree.links

    left = -1000
    width = 200

    group_outputs = nodes.new('NodeGroupOutput')
    group_outputs.location = (-left, 0)
    node_tree.outputs.new('NodeSocketGeometry', 'Out')

    grid = nodes.new(type='GeometryNodeMeshGrid')
    grid.location = (left + width, 100)
    grid.inputs['Vertices X'].default_value = resolution
    grid.inputs['Vertices Y'].default_value = resolution

    random = nodes.new(type='FunctionNodeRandomValue')
    random.data_type = 'FLOAT_VECTOR'
    random.inputs['Min'].default_value = [minimum] * 3
    random.inputs['Max'].default_value = [maximum] * 3
    random.location = (left + width, -200)

    sep = nodes.new(type='ShaderNodeSeparateXYZ')
    sep.location = (left + 2 * width, 0)
    links.new(random.outputs['Value'], sep.inputs['Vector'])

    comb = nodes.new(type='ShaderNodeCombineXYZ')
    comb.location = (left + 3 * width, 100)
    links.new(sep.outputs[0], comb.inputs[0])
    links.new(sep.outputs[1], comb.inputs[1])

    set_pos = nodes.new(type='GeometryNodeSetPosition')
    set_pos.location = (left + 3 * width, 400)
    links.new(grid.outputs['Mesh'], set_pos.inputs['Geometry'])
    links.new(comb.outputs[0], set_pos.inputs['Position'])

    less = nodes.new(type='ShaderNodeMath')
    less.label = 'PointControl'
    less.operation = "LESS_THAN"
    less.location = (left + 4 * width, 0)
    links.new(sep.outputs['Z'], less.inputs['Value'])

    mesh_sphere = nodes.new(type='GeometryNodeMeshUVSphere')
    mesh_sphere.location = (left + 2 * width, -200)
    mesh_sphere.inputs['Radius'].default_value = 0.05

    instance = nodes.new(type='GeometryNodeInstanceOnPoints')
    instance.location = (left + 5 * width, 200)

    links.new(less.outputs['Value'], instance.inputs['Selection'])
    links.new(set_pos.outputs['Geometry'], instance.inputs['Points'])
    links.new(mesh_sphere.outputs['Mesh'], instance.inputs['Instance'])

    mat = nodes.new(type='GeometryNodeSetMaterial')
    mat.location = (200, 0)
    mat.inputs[2].default_value = material

    links.new(instance.outputs['Instances'], mat.inputs[0])
    links.new(mat.outputs[0], group_outputs.inputs['Out'])

    return node_tree


def monte_carlo_nodes(material, radius=1, name="MonteCarloGroup"):
    """
    material is required to distinguish to states of spheres that are generated by the monte carlo simulation
    :param radius:
    :param material:
    :param name:
    :return:
    """
    node_tree = bpy.data.node_groups.new(name, type='GeometryNodeTree')

    nodes = node_tree.nodes
    links = node_tree.links

    left = -1000
    width = 200

    group_inputs = nodes.new('NodeGroupInput')
    group_inputs.location = (left, 0)

    group_outputs = nodes.new('NodeGroupOutput')
    group_outputs.location = (-left, 0)

    node_tree.inputs.new('NodeSocketGeometry', 'In')
    node_tree.outputs.new('NodeSocketGeometry', 'Out')

    mesh = nodes.new(type='GeometryNodeSubdivideMesh')
    mesh.location = (left + width, 100)
    mesh.inputs['Level'].default_value = 6
    links.new(group_inputs.outputs['In'], mesh.inputs['Mesh'])

    random = nodes.new(type='FunctionNodeRandomValue')
    random.data_type = 'FLOAT_VECTOR'
    random.inputs['Min'].default_value = Vector([-radius] * 3)
    random.inputs['Max'].default_value = Vector([radius] * 3)
    random.location = (left + width, -200)

    sep = nodes.new(type='ShaderNodeSeparateXYZ')
    sep.location = (left + 2 * width, 0)
    links.new(random.outputs['Value'], sep.inputs['Vector'])

    comb = nodes.new(type='ShaderNodeCombineXYZ')
    comb.location = (left + 3 * width, 100)
    links.new(sep.outputs[0], comb.inputs[0])
    links.new(sep.outputs[1], comb.inputs[1])

    set_pos = nodes.new(type='GeometryNodeSetPosition')
    set_pos.location = (left + 3 * width, 400)
    links.new(mesh.outputs['Mesh'], set_pos.inputs['Geometry'])
    links.new(comb.outputs[0], set_pos.inputs['Position'])

    less = nodes.new(type='ShaderNodeMath')
    less.label = 'PointControl'
    less.operation = "LESS_THAN"
    less.location = (left + 4 * width, 0)
    links.new(sep.outputs['Z'], less.inputs['Value'])

    mesh_sphere = nodes.new(type='GeometryNodeMeshUVSphere')
    mesh_sphere.location = (left + 2 * width, -200)
    mesh_sphere.inputs['Radius'].default_value = 0.05

    instance = nodes.new(type='GeometryNodeInstanceOnPoints')
    instance.location = (left + 5 * width, 200)

    links.new(less.outputs['Value'], instance.inputs['Selection'])
    links.new(set_pos.outputs['Geometry'], instance.inputs['Points'])
    links.new(mesh_sphere.outputs['Mesh'], instance.inputs['Instance'])

    mat = nodes.new(type='GeometryNodeSetMaterial')
    mat.location = (200, 0)
    mat.inputs[2].default_value = material

    links.new(instance.outputs['Instances'], mat.inputs[0])
    links.new(mat.outputs[0], group_outputs.inputs['Out'])

    return node_tree


def integral_checker(functions=[], colors=[], name="IntegralChecker", mesh_points=100, domain=[0, 1]):
    """
       create a group that converts a plane to align its lower vertices with the first function and its
       upper vertices to align with the second function

       the faces are colored alternatingly

       a subdivision node is retrievable for dynamics

       :param material:
       :param name:
       :return:
       """
    node_tree = bpy.data.node_groups.new(name, type='GeometryNodeTree')

    nodes = node_tree.nodes
    links = node_tree.links

    left = -1000
    width = 200

    group_inputs = nodes.new('NodeGroupInput')
    group_inputs.location = (left, 0)
    group_outputs = nodes.new('NodeGroupOutput')
    group_outputs.location = (-left, 0)

    node_tree.inputs.new('NodeSocketGeometry', 'In')
    node_tree.outputs.new('NodeSocketGeometry', 'Out')
    node_tree.outputs.new('NodeSocketColor', 'FaceColorInformation')

    # input data nodes
    pos = nodes.new(type="GeometryNodeInputPosition")
    pos.location = (left + width, -100)
    pos.hide = True

    index = nodes.new(type="GeometryNodeInputIndex")
    index.location = (left + width, -200)
    index.hide = True

    nIntervals = nodes.new(type="ShaderNodeValue")
    nIntervals.name = "nIntervals"
    nIntervals.label = 'nIntervals'
    nIntervals.location = (left + width, -300)
    nIntervals.outputs[0].default_value = 10
    nIntervals.hide = True

    spacing = create_group_from_vector_function(nodes, [str(domain[1] - domain[0]) + ',n,/'],
                                                scalar_parameters=['n'], name='spacing',
                                                node_group_type='Geometry')
    spacing.location = (left + 2 * width, -100)
    spacing.hide = True
    length = 3

    links.new(nIntervals.outputs[0], spacing.inputs['n'])

    # discretize function
    # replace x-> x,1-,d,/,floor,d,*,1+
    u = domain[0]
    functions[0][1] = functions[0][1].replace('x', 'x,' + str(u) + ',-,d,/,round,d,*,' + str(u) + ',+')
    functions[1][1] = functions[1][1].replace('x', 'x,' + str(u) + ',-,d,/,round,d,*,' + str(u) + ',+')
    func1 = create_group_from_vector_function(nodes, functions=functions[0], scalar_parameters=['d'],
                                              node_group_type='Geometry', name='f')
    func1.location = (left + length * width, -200)
    func1.hide = True

    func2 = create_group_from_vector_function(nodes, functions=functions[1], scalar_parameters=['d'],
                                              node_group_type='Geometry', name='g')
    func2.location = (left + length * width, -300)
    func2.hide = True
    links.new(spacing.outputs[0], func1.inputs['d'])
    links.new(spacing.outputs[0], func2.inputs['d'])
    length += 1

    ifnode = if_node(nodes, 'y,0,<', parameters=['In', 'True', 'False'], name='ifNode', node_type='Geometry')
    ifnode.location = (left + length * width, -250)
    ifnode.hide = True
    length += 1

    set_pos = nodes.new(type='GeometryNodeSetPosition')
    set_pos.location = (left + length * width, 100)
    length += 1
    links.new(group_inputs.outputs[0], set_pos.inputs['Geometry'])
    links.new(set_pos.outputs['Geometry'], group_outputs.inputs[0])

    links.new(pos.outputs[0], func1.inputs['In'])
    links.new(pos.outputs[0], func2.inputs['In'])

    links.new(pos.outputs[0], ifnode.inputs['In'])
    links.new(func1.outputs[0], ifnode.inputs['False'])
    links.new(func2.outputs[0], ifnode.inputs['True'])
    links.new(ifnode.outputs[0], set_pos.inputs['Position'])
    # checker color faces
    checker = create_group_from_vector_function(nodes, functions=[
        str(mesh_points / 2) + ",n,/,idx,+,n,*," + str(2 * mesh_points) + ",/,1,%,round"],
                                                scalar_parameters=['idx', 'n'],
                                                name='checkerFunction', node_group_type='Geometry')
    checker.location = (left + 3 * width, 300)
    links.new(index.outputs[0], checker.inputs['idx'])
    links.new(nIntervals.outputs[0], checker.inputs['n'])

    ramp = nodes.new(type='ShaderNodeValToRGB')
    ramp.location = (left + 5 * width, 200)
    ramp.color_ramp.elements[0].color = get_color_from_string(colors[0])
    ramp.color_ramp.elements[1].color = get_color_from_string(colors[1])
    links.new(checker.outputs[0], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], group_outputs.inputs['FaceColorInformation'])
    node_tree.outputs['FaceColorInformation'].attribute_domain = 'FACE'

    return node_tree


##########################
## Penrose tilings I
############################
def create_hexagon_tilings(name='HexagonTilings', material='drawing', level=1, scale=10, **kwargs):
    """
       create a group that places hexagons at each vertex point of the mesh
       a subdivision node is retrievable for dynamics

       :return:
    """

    material = get_material(material, **kwargs)
    node_tree = bpy.data.node_groups.new(name, type='GeometryNodeTree')

    nodes = node_tree.nodes
    links = node_tree.links

    left = -1000
    width = 200
    length = 0

    group_inputs = nodes.new('NodeGroupInput')
    group_inputs.location = (left, 0)
    group_outputs = nodes.new('NodeGroupOutput')
    group_outputs.location = (-left, 0)
    length += 1

    make_new_socket(node_tree, name='In', io='INPUT', type='NodeSocketGeometry')
    make_new_socket(node_tree, name='Out', io='OUTPUT', type='NodeSocketGeometry')

    # input data nodes
    pos = nodes.new(type="GeometryNodeInputPosition")
    pos.location = (left + length * width, -100)
    pos.hide = True

    index = nodes.new(type="GeometryNodeInputIndex")
    index.location = (left + length * width, -200)
    index.hide = True
    length += 1

    # map position of the vertex into the hexagonal grid
    r3 = np.sqrt(3)
    r32 = r3 / 2
    mapping = create_group_from_vector_function(nodes,
                                                functions=[str(r3) + ',x,*,' + str(r32) + ',y,*,+', '1.5,y,*', 'z'],
                                                node_group_type='Geometry', name='HexagonalMapping')
    mapping.location = (left + length * width, -100)
    length += 1

    # scaling

    scaling = nodes.new(type="ShaderNodeVectorMath")
    scaling.name = 'scale'
    scaling.label = 'scale'
    scaling.location = (left + length * width, -100)
    scaling.operation = 'SCALE'
    scaling.inputs['Scale'].default_value = scale
    links.new(mapping.outputs['Out'], scaling.inputs[0])
    length += 1

    links.new(pos.outputs['Position'], mapping.inputs['In'])

    sub_div = nodes.new(type='GeometryNodeSubdivideMesh')
    sub_div.location = (left + length * width, 100)
    sub_div.inputs['Level'].default_value = level
    links.new(group_inputs.outputs[0], sub_div.inputs[0])
    length += 1

    set_pos = nodes.new(type='GeometryNodeSetPosition')
    set_pos.location = (left + length * width, 100)
    links.new(sub_div.outputs[0], set_pos.inputs['Geometry'])
    links.new(scaling.outputs[0], set_pos.inputs['Position'])

    # create hexagon

    hexagon = nodes.new(type='GeometryNodeMeshCylinder')
    hexagon.location = (left + length * width, -400)
    length += 1
    hexagon.inputs['Vertices'].default_value = 6
    hexagon.inputs['Depth'].default_value = 0.1

    instance = nodes.new(type='GeometryNodeInstanceOnPoints')
    instance.location = (left + length * width, 200)
    instance.inputs['Rotation'].default_value = [0, 0, np.pi / 6]
    length += 1

    # material

    mat = nodes.new(type='GeometryNodeSetMaterial')
    mat.location = (left + length * width, 200)
    mat.inputs[2].default_value = material
    length += 1

    # realize instances
    real = nodes.new(type='GeometryNodeRealizeInstances')
    real.location = (left + length * width, 200)
    length += 1

    # set shade smooth
    smooth = nodes.new(type='GeometryNodeSetShadeSmooth')
    smooth.location = (left + length * width, 200)
    length += 1
    smooth.domain = 'FACE'

    links.new(set_pos.outputs['Geometry'], instance.inputs['Points'])
    links.new(hexagon.outputs['Mesh'], instance.inputs['Instance'])
    links.new(instance.outputs['Instances'], mat.inputs['Geometry'])
    links.new(mat.outputs['Geometry'], real.inputs['Geometry'])
    links.new(real.outputs['Geometry'], smooth.inputs['Geometry'])
    links.new(smooth.outputs['Geometry'], group_outputs.inputs[0])

    return node_tree


def create_z3(name='Z3_generator', n=3, base="PENROSE", **kwargs):
    """
    create a geometry node that generates Z5 tuples

    :param n:
    :param name:
    :param level: 3 mean that the tuples range from (-3,-3,-3,-3,-3) to (3,3,3,3,3)
    :return:
    """
    node_tree = bpy.data.node_groups.new(name, type='GeometryNodeTree')

    nodes = node_tree.nodes
    links = node_tree.links

    left = -1000
    width = 200
    length = 0

    group_outputs = nodes.new('NodeGroupOutput')
    group_outputs.location = (-3 * left, 0)
    length += 1

    make_new_socket(node_tree, name='Geometry', io='OUTPUT', type='NodeSocketGeometry')

    # create point cloud
    n_value = nodes.new(type="ShaderNodeValue")
    n_value.name = 'range'
    n_value.label = 'range'
    n_value.outputs[0].default_value = n
    n_value.location = (left + length * width, -150)
    n_value.hide = True

    # create rotation angles
    tl = -20
    th = -2
    thetas = []
    t_labels = ['t_' + str(i) for i in range(3)]
    for i in range(3):
        thetas.append(
            InputValue(node_tree, location=(tl, th + 2 - 0.5 * i), value=0, name=t_labels[i], label=t_labels[i]))
    tl += 1
    rotations = []
    dirs = choose(list(range(3)), 2)
    orientation = [1, -1, 1]  # account for  right-handedness of the rotation with respect to Euler angles
    for i in range(3):
        rotations.append(Rotation(node_tree, location=(tl, th + 1 - 0.4 * i),
                                  angle=thetas[2 - i].std_out, u=dirs[i][0], v=dirs[i][1],
                                  dimension=3, name='R_' + str(i), label='R_' + str(i), orientation=orientation[i],
                                  hide=True)
                         )

    tl += 1

    # create rotation marker
    ml = tl + 1
    mh = 10

    join_marker = JoinGeometry(node_tree, location=(ml + 4, mh / 2))

    for i in range(3):
        grid = Grid(node_tree, location=(ml, mh - i), vertices_x=2, vertices_y=2, hide=True)
        wire = WireFrame(node_tree, location=(ml + 0.5, mh + 0.25 - i),
                         radius=0.5, hide=True)
        local_join = JoinGeometry(node_tree, location=(ml + 1, mh + 0.25 - i), hide=True)
        mat = SetMaterial(node_tree, location=(ml + 1, mh - i),
                          material='x' + str(dirs[i][0] + 1) + str(dirs[i][1] + 1) + "_color", hide=True)
        comb_xyz = node_tree.nodes.new(type='ShaderNodeCombineXYZ')
        comb_xyz.location = ((ml + 1) * 200, (mh - 0.5 - i) * 200)
        comb_xyz.inputs['X'].default_value = np.pi / 2
        comb_xyz.hide = True
        node_tree.links.new(thetas[2 - i].std_out, comb_xyz.inputs['Y'])
        trans = TransformGeometry(node_tree, location=(ml + 2, mh - i),
                                  translation=[-6 + 0.88889 * i, 0, 2.75],
                                  rotation=comb_xyz.outputs['Vector'], scale=[0.05] * 3, hide=True)

        create_geometry_line(node_tree, [grid, wire, local_join, trans, join_marker])
        create_geometry_line(node_tree, [grid, mat, local_join])

    # create switch for rotation markers
    switch = InputBoolean(node_tree, location=(ml + 4, mh / 2 - 1), name='RotationPanelSwitch', value=True)
    del_rot_marker = DeleteGeometry(node_tree, location=(ml + 5, mh / 2), name="DeleteRotationMarker",
                                    selection=switch.std_out)
    create_geometry_line(node_tree, [join_marker, del_rot_marker])

    # create the basis vectors
    basis_labels = ["u", "v", "n"]
    input_vecs = []
    if base == "STANDARD":
        u = [1, 0, 0]
        v = [0, 1, 0]
        n = [0, 0, 1]
    elif base == "PENROSE":
        r3 = np.sqrt(3)
        u = [1 / 6 * (3 + r3), 1 / 6 * (-3 + r3), -1 / r3]
        v = [1 / 6 * (-3 + r3), 1 / 6 * (3 + r3), -1 / r3]
        n = [1 / r3, 1 / r3, 1 / r3]
    basis = [u, v, n]
    for i, name in enumerate(basis_labels):
        value = Vector(basis[i])
        input_vec = InputVector(
            node_tree,
            location=(tl, -1.5 - 0.25 * i),
            value=value,
            name=name,
            label=name,
            hide=True
        )
        input_vecs.append(input_vec)

    tl += 1

    # create 9 linear maps to rotate the basis vectors
    rot_labels = ["rotU", "rotV", "rotN"]
    input_vec_sockets = []
    for i in range(0, 3):
        last_vec_socket = input_vecs[i].std_out
        for j in range(3):
            lm = LinearMap(node_tree, location=(tl + j, th - 0.4 * i), dimension=3,
                           name="LinearMap" + str(i) + str(2 - j), label=rot_labels[i] + "_" + str(2 - j), hide=True)
            for k in range(3):
                node_tree.links.new(rotations[2 - j].outputs[k], lm.inputs[k + 1])
            node_tree.links.new(last_vec_socket, lm.inputs[4])
            last_vec_socket = lm.outputs[0]  # couple linear map with last one
        input_vec_sockets += [last_vec_socket]
    # create sigmas
    sigma_node = InputVector(node_tree, location=(-5 + length, -4.25),
                             value=Vector([0.2] * 3),
                             name='sigma', label='sigma', hide=True)

    length += 1

    two_n_plus_one = nodes.new(type='ShaderNodeMath')
    two_n_plus_one.operation = "MULTIPLY_ADD"
    two_n_plus_one.location = (left + length * width, -150)
    two_n_plus_one.inputs[1].default_value = 2  # Multiplier
    two_n_plus_one.inputs[2].default_value = 1  # Addend
    links.new(n_value.outputs['Value'], two_n_plus_one.inputs['Value'])
    length += 1

    power = nodes.new(type='ShaderNodeMath')
    power.operation = "POWER"
    power.location = (left + length * width, 0)
    power.inputs[1].default_value = 3  # Exponent
    links.new(two_n_plus_one.outputs['Value'], power.inputs[0])
    length += 1

    line = []

    points = Points(node_tree, location=(-5 + length, 1.5), radius=0.01,
                    count=power.outputs['Value'])
    line.append(points)

    index = Index(node_tree, location=(-5 + (length - 1), -1.5))

    index2tuple = make_function(nodes, functions={
        "P": ['index,base,2,**,/,floor,range,-', 'index,base,2,**,%,base,/,floor,range,-', 'index,base,%,range,-']
    }, name="Index2Tuple", inputs=["index", "base", "range"], outputs=["P"], scalars=["index", "base", "range"],
                                vectors=["P"])
    index2tuple.location = (left + length * width, -300)
    links.new(index.outputs[0], index2tuple.inputs['index'])
    links.new(two_n_plus_one.outputs[0], index2tuple.inputs['base'])
    links.new(n_value.outputs[0], index2tuple.inputs['range'])

    length += 1

    # project to the orthogonal 3 space
    ortho_projection_pp = make_function(node_tree.nodes, functions={
        'Vector': ['0', '0', 'n,P,sigma,sub,dot']
    }, name='OrthgonalProjector',
                                        inputs=["n", "sigma"] + ["P"], outputs=['Vector'],
                                        vectors=["n", "P", 'sigma', "Vector"])
    ortho_projection_pp.location = (left + length * 200, 0)
    ortho_projection_pp.hide = True
    node_tree.links.new(input_vec_sockets[2], ortho_projection_pp.inputs["n"])
    node_tree.links.new(index2tuple.outputs["P"], ortho_projection_pp.inputs["P"])
    node_tree.links.new(sigma_node.std_out, ortho_projection_pp.inputs['sigma'])

    # project to tiling plane
    para_projection_pp = make_function(node_tree.nodes, functions={
        'Vector': ['u,P,sigma,sub,dot', 'v,P,sigma,sub,dot', '0']
    }, name='ParaProjector', inputs=["u", "v", "sigma", "P"], outputs=['Vector'],
                                       vectors=["u", "v", "sigma", "P", "Vector"])
    para_projection_pp.location = (left + length * 200, -50)
    para_projection_pp.hide = True
    node_tree.links.new(input_vec_sockets[0], para_projection_pp.inputs["u"])
    node_tree.links.new(input_vec_sockets[1], para_projection_pp.inputs["v"])
    node_tree.links.new(index2tuple.outputs["P"], para_projection_pp.inputs["P"])
    node_tree.links.new(sigma_node.std_out, para_projection_pp.inputs["sigma"])

    length += 1

    # create attributes
    attr = StoredNamedAttribute(node_tree, location=(-5 + length, 1.5),
                                data_type='FLOAT_VECTOR',
                                domain='POINT', name=name,
                                value=index2tuple.outputs["P"])
    line.append(attr)
    length += 1

    ico_sphere = IcoSphere(node_tree, location=(-5 + length, 2),
                           radius=0.025, subdivisions=2, hide=True)

    set_pos = SetPosition(node_tree, location=(-5 + length, 1.5),
                          position=ortho_projection_pp.outputs['Vector'],
                          label="ProjectionOrtho"
                          )
    line.append(set_pos)
    length += 1

    del_geo_3d = DeleteGeometry(node_tree, location=(-5 + length, 2))
    line.append(del_geo_3d)
    length += 1

    ico_instance = InstanceOnPoints(node_tree, location=(-5 + length, 1.5), instance=ico_sphere.geometry_out,
                                    hide=True)
    ortho_projection_scale = InputValue(node_tree, location=(-5 + length, 1.25), name="OrthoProjectionScale", value=0)
    shift2 = TransformGeometry(node_tree, location=(-4 + length, 1.5), translation=Vector([-2.5, 0, 1]),
                               scale=ortho_projection_scale.std_out)
    line.append(ico_instance)
    line.append(shift2)

    # 2D line
    line2 = []
    line2.append(points)

    set_pos2 = SetPosition(node_tree, location=(-5 + length, 0.5),
                           position=para_projection_pp.outputs['Vector'],
                           label="Projection2D")

    line2.append(set_pos2)
    length += 1

    stored_index = StoredNamedAttribute(node_tree, location=(-5 + length, 0.5),
                                        name="SavedIndex",
                                        data_type='INT',
                                        value=index.std_out)
    line2.append(stored_index)
    length += 1

    # create 8 points the form the convex hull
    nl = 0
    nh = -3

    points8 = Points(node_tree, location=(nl, nh), count=8)
    idx = Index(node_tree, location=(nl - 1, nh - 1))

    # this function works also for indices with an arbitrary offset. We don't have to worry,\
    # when the index doesn't start at 0
    mod = make_function(node_tree.nodes, functions={
        'p': ['index,8,%,4,/,floor', 'index,4,%,2,/,floor', 'index,2,%'],
    }, name="Mod8", inputs=['index'], outputs=['p'], scalars=['index'], vectors=['p'])
    node_tree.links.new(idx.std_out, mod.inputs['index'])
    mod.location = (nl * 200, nh * 200 - 200)
    mod.hide = True
    nl += 1

    nline = [points8]
    # create attributes
    attr = StoredNamedAttribute(node_tree, location=(nl, nh),
                                data_type='FLOAT_VECTOR', domain='POINT',
                                name=name, value=mod.outputs["p"])
    nline.append(attr)
    nl += 1

    projection8 = make_function(node_tree.nodes, functions={
        'Vector': ['0', '0', 'n,p,dot']
    }, name='OrthgonalProjector',
                                inputs=["n", "p"], outputs=['Vector'], vectors=["n", "p"] + ['Vector'])
    projection8.location = (nl * 200, (nh - 1) * 200)
    projection8.hide = True

    node_tree.links.new(input_vec_sockets[2], projection8.inputs["n"])
    node_tree.links.new(mod.outputs["p"], projection8.inputs["p"])

    set_pos8 = SetPosition(node_tree, location=(nl, nh),
                           position=projection8.outputs['Vector']
                           )
    nline.append(set_pos8)
    nl += 1
    hull = ConvexHull(node_tree, location=(nl, nh))
    hull_wire = WireFrame(node_tree, location=(nl + 1, nh + 0.25), radius=0.01)
    hull_mat = SetMaterial(node_tree, location=(nl + 3, nh), material='plastic_joker')
    ico = IcoSphere(node_tree, location=(nl, nh - 1), subdivisions=2, radius=0.05)
    iop = InstanceOnPoints(node_tree, location=(nl + 1, nh), instance=ico.geometry_out)
    ch_scale = InputValue(node_tree, location=(nl + 1, nh - 0.25), name='ConvexHullScale', value=0)
    shift = TransformGeometry(node_tree, location=(nl + 4, nh), translation=Vector([-2.5, 0, 1]),
                              scale=ch_scale.std_out)
    hull_join = JoinGeometry(node_tree, location=(nl + 2, nh))
    create_geometry_line(node_tree, [hull, hull_wire, hull_join])
    nline += [hull, iop, hull_join, hull_mat, shift]
    nl += 2

    # convex hull test

    convex_hull_test = InsideConvexHull3D(node_tree, location=(1, -1.25),
                                          target_geometry=hull.geometry_out,
                                          source_position=ortho_projection_pp.outputs['Vector']
                                          )

    node_tree.links.new(convex_hull_test.std_out, set_pos2.node.inputs['Selection'])
    node_tree.links.new(convex_hull_test.std_out, set_pos.node.inputs['Selection'])

    del_geo = DeleteGeometry(node_tree, location=(-5 + length, 0),
                             selection=convex_hull_test.outputs['Is Outside'])
    links.new(convex_hull_test.outputs['Is Outside'], del_geo_3d.inputs['Selection'])

    line2.append(del_geo)

    # create faces in all three orientations for each selected point
    # and pick only those that completely lie inside the convex hull
    # the face directions are stored in two components of an input vector

    ml = 2
    mh = -0.5

    input_dirs = []
    for i, dir in enumerate(dirs):
        input_dirs.append(InputVector(node_tree, location=(ml, mh - 0.25 * i),
                                      value=dir + [0],
                                      name=str(dir), label=str(dir), hide=True)
                          )

    ml += 1
    mh -= 1

    sel_index = NamedAttribute(node_tree, location=(ml, mh),
                               data_type='INT',
                               name="SavedIndex")

    ml += 1

    sel_index2tuple = make_function(nodes, functions={
        "P": ['index,base,2,**,/,floor,range,-', 'index,base,2,**,%,base,/,floor,range,-', 'index,base,%,range,-']
    }, name="SelectedIndex2Tuple", inputs=["index", "base", "range"], outputs=["P"], scalars=["index", "base", "range"],
                                    vectors=["P"])
    sel_index2tuple.location = (ml * width, mh * width)
    links.new(sel_index.std_out, sel_index2tuple.inputs['index'])
    links.new(two_n_plus_one.outputs[0], sel_index2tuple.inputs['base'])
    links.new(n_value.outputs[0], sel_index2tuple.inputs['range'])

    ml += 1

    line3 = []
    sub_join = JoinGeometry(node_tree, location=(nl + 4, 0))

    scale_2d = InputValue(node_tree, location=(nl + 4, -0.25), name='ScaleProjection', value=0)
    scale_elements = ScaleElements(node_tree, location=(nl + 5, 0),
                                   scale=scale_2d.std_out)

    if 'final_translation' in kwargs:
        translation = kwargs.pop('final_translation')
    else:
        translation = Vector()
    if 'final_rotation' in kwargs:
        rotation = kwargs.pop('final_rotation')
    else:
        rotation = Vector()
    if 'final_scale' in kwargs:
        scale = kwargs.pop('final_scale')
    else:
        scale = Vector([1, 1, 1])
    transform = TransformGeometry(node_tree, location=(nl + 6, 0),
                                  translation=translation, rotation=rotation, scale=scale)

    line3.append(sub_join)
    line3.append(scale_elements)
    line3.append(transform)
    join = JoinGeometry(node_tree, location=(nl + 7, 0))
    nline.append(join)
    line.append(join)
    # line2.append(join)
    line3.append(join)
    create_geometry_line(node_tree, nline)
    create_geometry_line(node_tree, line)
    create_geometry_line(node_tree, line2)
    create_geometry_line(node_tree, line3)

    create_geometry_line(node_tree, [del_rot_marker, join])
    links.new(join.geometry_out, group_outputs.inputs['Geometry'])

    # make face generation groups

    ml += 1
    for i in range(3):
        direction = str(dirs[i][0] + 1) + str(dirs[i][1] + 1)
        face_generation_group = make_face_generation_group_3d(nodes, basis_labels, ["sigma"],
                                                              material='x' + direction + "_color",
                                                              name="FaceGenerator" + str(dirs[i]))
        face_generation_group.location = ((ml + 0.25 * i) * 200, (mh - 0.25 * i) * 200)
        face_generation_group.hide = True

        node_tree.links.new(del_geo.geometry_out, face_generation_group.inputs['Geometry'])
        node_tree.links.new(input_dirs[i].std_out, face_generation_group.inputs['Direction'])
        node_tree.links.new(sel_index2tuple.outputs["P"], face_generation_group.inputs['P'])
        node_tree.links.new(face_generation_group.outputs["Geometry"], sub_join.inputs['Geometry'])

        for j, label in enumerate(basis_labels):
            node_tree.links.new(input_vec_sockets[j], face_generation_group.inputs[label])

        node_tree.links.new(sigma_node.std_out, face_generation_group.inputs["sigma"])
        node_tree.links.new(hull.geometry_out, face_generation_group.inputs['Convex Hull'])

    # 3d visuals

    # adjusted normal, since the thetas are switched
    combine = CombineXYZ(node_tree, location=(-11 + length, 3.5), x=thetas[0].std_out, y=thetas[1].std_out,
                         z=thetas[2].std_out,
                         name="CombineEulerAngles")

    join3d = JoinGeometry(node_tree, location=(length - 1, 5))
    # grid
    grow_scale = InputValue(node_tree, (-7 + length, 5), value=0, name="GrowthScale3D")
    position = Position(node_tree, (-7 + length, 4.75), name="CubiePosition")
    grow_grid = make_function(node_tree, functions={
        'selection': 'pos,length,r,<'
    }, inputs=['pos', 'r'], outputs=['selection'], scalars=['r', 'selection'], vectors=['pos'], name='GrowFunction',
                              hide=True)
    grow_grid.location = ((-6 + length) * 200, 5 * 200)
    node_tree.links.new(grow_scale.std_out, grow_grid.inputs['r'])
    node_tree.links.new(position.std_out, grow_grid.inputs['pos'])
    points3d = Points(node_tree, location=(-5 + length, 5), count=power.outputs['Value'])
    cubies = CubeMesh(node_tree, location=(-5 + length, 4), size=[0.1, 0.1, 0.1], name="Cubies")
    iop3d = InstanceOnPoints(node_tree, location=(-4 + length, 5), instance=cubies.geometry_out,
                             selection=grow_grid.outputs['selection'])
    set_pos3d = SetPosition(node_tree, location=(-3 + length, 5), position=index2tuple.outputs["P"], name="Pos3D")
    create_geometry_line(node_tree, [points3d, set_pos3d, iop3d, join3d])

    # selected
    select_scale = InputValue(node_tree, location=(-5 + length, 6.75), value=0, name='SelectedScale')
    sel_icos = IcoSphere(node_tree, location=(-5 + length, 7), radius=select_scale.std_out, subdivisions=2)
    set_pos3d_sel = SetPosition(node_tree, location=(-4 + length, 7),
                                selection=convex_hull_test.outputs["Is Inside"],
                                position=index2tuple.outputs["P"], name="Pos3DSelected")
    stored_attr = StoredNamedAttribute(node_tree, location=(-3 + length, 7), data_type='INT',
                                       name='SavedIndex2', value=index.std_out)
    del_geo_3d_sel = DeleteGeometry(node_tree, location=(-2 + length, 7),
                                    selection=convex_hull_test.outputs["Is Outside"])
    iop_sel = InstanceOnPoints(node_tree, location=(-1 + length, 6.75), instance=sel_icos.geometry_out)
    sel_mat = SetMaterial(node_tree, location=(length, 7), material="plastic_example")
    create_geometry_line(node_tree, [points3d, set_pos3d_sel, stored_attr, del_geo_3d_sel, iop_sel, sel_mat, join3d])

    # plane
    scale_plane = InputValue(node_tree, (-6 + length, 2.75), name='PlaneScale', value=0)
    mesh = Grid(node_tree, location=(-5 + length, 3), size_x=scale_plane.std_out, size_y=scale_plane.std_out,
                vertices_x=11, vertices_y=11)

    rotate = TransformGeometry(node_tree, location=(-4 + length, 3), name="Rotation", rotation=combine.std_out)
    wire = WireFrame(node_tree, geometry=mesh.geometry_out, location=(-3 + length, 3))
    mesh_mat = SetMaterial(node_tree, location=(-2 + length, 3), material='plastic_custom1')
    create_geometry_line(node_tree, [mesh, rotate, wire, mesh_mat, join3d])

    # ortho line
    scale_ortho_line = InputValue(node_tree, (-6 + length, 2.25), name='OrthoLineScale', value=0)
    line = MeshLine(node_tree, location=(-5 + length, 2.5), count=10, start_location=Vector([0, 0, -6]),
                    end_location=Vector([0, 0, 6]), )
    rotate_line = TransformGeometry(node_tree, location=(-4 + length, 2.5), name="RotationLine",
                                    scale=scale_ortho_line.std_out, rotation=combine.std_out)
    wire_line = WireFrame(node_tree, geometry=line.geometry_out, location=(-3 + length, 2.5))
    mesh_mat_line = SetMaterial(node_tree, location=(-2 + length, 2.5), material='plastic_joker')
    create_geometry_line(node_tree, [line, rotate_line, wire_line, mesh_mat_line, join3d])

    # voronoi
    scale_voronoi = InputValue(node_tree, location=(-6 + length, 5.75), name="VoronoiScale", value=0)
    voronoi = CubeMesh(node_tree, location=(-5 + length, 6), name="Voronoi")
    voronoi_shift = TransformGeometry(node_tree, location=(-4 + length, 6), translation=[0.5] * 3,
                                      scale=scale_voronoi.std_out)
    voronoi_wire = WireFrame(node_tree, location=(-3 + length, 6))
    voronoi_mat = SetMaterial(node_tree, location=(-2 + length, 6), material='plastic_drawing')
    create_geometry_line(node_tree, [voronoi, voronoi_shift, voronoi_wire, voronoi_mat, join3d])

    sel_index_3d = NamedAttribute(node_tree, location=(-10 + length, 7),
                                  data_type='INT',
                                  name="SavedIndex2")

    ml += 1

    sel_index2tuple_3d = make_function(nodes, functions={
        "P": ['index,base,2,**,/,floor,range,-', 'index,base,2,**,%,base,/,floor,range,-', 'index,base,%,range,-']
    }, name="SelectedIndex2Tuple", inputs=["index", "base", "range"], outputs=["P"], scalars=["index", "base", "range"],
                                       vectors=["P"])
    sel_index2tuple_3d.location = ((-9 + length) * width, 1400)
    links.new(sel_index_3d.std_out, sel_index2tuple_3d.inputs['index'])
    links.new(two_n_plus_one.outputs[0], sel_index2tuple_3d.inputs['base'])
    links.new(n_value.outputs[0], sel_index2tuple_3d.inputs['range'])

    # show faces

    face_scale = InputValue(node_tree, location=(-6 + length, 9), name='FaceScale', value=0)
    for i in range(3):
        direction = str(dirs[i][0] + 1) + str(dirs[i][1] + 1)
        face_generation_group = make_face_generation_group_3d_unprojected(nodes, basis_labels, ["sigma"],
                                                                          material='x' + direction + "_color",
                                                                          name="FaceGenerator" + str(dirs[i]))
        face_generation_group.location = ((-5 + length) * 200, (8.5 - 0.5 * i) * 200)
        face_generation_group.hide = True

        node_tree.links.new(del_geo_3d_sel.geometry_out, face_generation_group.inputs['Geometry'])
        node_tree.links.new(face_scale.std_out, face_generation_group.inputs['FaceScale'])
        node_tree.links.new(input_dirs[i].std_out, face_generation_group.inputs['Direction'])
        node_tree.links.new(sel_index2tuple_3d.outputs["P"], face_generation_group.inputs['P'])
        node_tree.links.new(face_generation_group.outputs["Geometry"], join3d.inputs['Geometry'])

        comps_u = [0, 0, 0]
        comps_u[dirs[i][0]] = 1
        u = Vector(comps_u)
        comps_v = [0, 0, 0]
        comps_v[dirs[i][1]] = 1
        v = Vector(comps_v)
        face_generation_group.inputs['u'].default_value = u
        face_generation_group.inputs['v'].default_value = v
        node_tree.links.new(input_vec_sockets[2], face_generation_group.inputs["n"])

        node_tree.links.new(sigma_node.std_out, face_generation_group.inputs["sigma"])
        node_tree.links.new(hull.geometry_out, face_generation_group.inputs['Convex Hull'])

    # connect to remaining geometry
    view_rotation = InputVector(node_tree, location=(length - 1, 0.75), value=Vector(), name='ViewRotation')
    transform_3d = TransformGeometry(node_tree, location=(length, 1), translation=Vector([0, 0, 0]),
                                     rotation=view_rotation.std_out,
                                     scale=[0.25] * 3, name="Transform3D")
    create_geometry_line(node_tree, [join3d, transform_3d, join])
    return node_tree


def create_z5(name='Z5_generator', n=3, base="PENROSE", **kwargs):
    """
    create a geometry node that generates Z5 tuples

    :param n:
    :param name:
    :param level: 3 mean that the tuples range from (-3,-3,-3,-3,-3) to (3,3,3,3,3)
    :return:
    """
    node_tree = bpy.data.node_groups.new(name, type='GeometryNodeTree')

    nodes = node_tree.nodes
    links = node_tree.links

    left = -1000
    width = 200
    length = 0

    group_outputs = nodes.new('NodeGroupOutput')
    group_outputs.location = (-3 * left, 0)
    length += 1

    make_new_socket(node_tree, name='Geometry', io='OUTPUT', type='NodeSocketGeometry')

    # create point cloud
    n_value = nodes.new(type="ShaderNodeValue")
    n_value.name = 'range'
    n_value.label = 'range'
    n_value.outputs[0].default_value = n
    n_value.location = (left + length * width, -150)
    n_value.hide = True

    # create rotation angles
    tl = -20
    th = -2
    thetas = []
    t_labels = ['t_' + str(i) for i in range(10)]
    for i in range(10):
        thetas.append(
            InputValue(node_tree, location=(tl, th + 2 - 0.5 * i), value=0, name=t_labels[i], label=t_labels[i]))
    tl += 1
    rotations = []
    dirs = choose(list(range(5)), 2)
    for i in range(10):
        rotations.append(Rotation(node_tree, location=(tl, th + 1 - 0.4 * i),
                                  angle=thetas[i].std_out, u=dirs[i][0], v=dirs[i][1],
                                  dimension=5, name='R_' + str(i), label='R_' + str(i), hide=True)
                         )

    tl += 1

    # create rotation marker
    ml = tl + 1
    mh = 10

    join_marker = JoinGeometry(node_tree, location=(ml + 4, mh / 2))

    for i in range(10):
        grid = Grid(node_tree, location=(ml, mh - i), vertices_x=2, vertices_y=2, hide=True)
        wire = WireFrame(node_tree, location=(ml + 0.5, mh + 0.25 - i),
                         radius=0.5, hide=True)
        local_join = JoinGeometry(node_tree, location=(ml + 1, mh + 0.25 - i), hide=True)
        mat = SetMaterial(node_tree, location=(ml + 1, mh - i),
                          material='x' + str(dirs[i][0] + 1) + str(dirs[i][1] + 1) + "_color", hide=True)
        comb_xyz = node_tree.nodes.new(type='ShaderNodeCombineXYZ')
        comb_xyz.location = ((ml + 1) * 200, (mh - 0.5 - i) * 200)
        comb_xyz.inputs['X'].default_value = np.pi / 2
        comb_xyz.hide = True
        node_tree.links.new(thetas[i].std_out, comb_xyz.inputs['Y'])
        trans = TransformGeometry(node_tree, location=(ml + 2, mh - i),
                                  translation=[-6 + 0.88889 * i, 0, 2.75],
                                  rotation=comb_xyz.outputs['Vector'], scale=[0.05] * 3, hide=True)

        create_geometry_line(node_tree, [grid, wire, local_join, trans, join_marker])
        create_geometry_line(node_tree, [grid, mat, local_join])

    # create the basis vectors
    basis_labels = ["u", "U", "v", "V", "n1", "m1", "n2", "m2", "n3", "m3"]
    ins = basis_labels[4:]
    input_vecs = []
    if base == "STANDARD":
        u_5d = [1, 0, 0, 0, 0]
        v_5d = [0, 1, 0, 0, 0]
        n1_5d = [0, 0, 1, 0, 0]
        n2_5d = [0, 0, 0, 1, 0]
        n3_5d = [0, 0, 0, 0, 1]
    elif base == "PENROSE":
        r5 = np.sqrt(5)
        r2 = np.sqrt(2)
        r10 = r2 * r5
        u_5d = [-1 / 2 * np.sqrt(3 / 5 + 1 / r5), (-1 + r5) / 2 / r10, (-1 + r5) / 2 / r10,
                -1 / 2 * np.sqrt(3 / 5 + 1 / r5), r2 / r5]
        v_5d = [-1 / 2 * np.sqrt(1 - 1 / r5), 1 / 2 * np.sqrt(1 + 1 / r5), -1 / 2 * np.sqrt(1 + 1 / r5),
                1 / 2 * np.sqrt(1 - 1 / r5), 0]
        n1_5d = [(-1 + r5) / 2 / r10, -1 / 2 * np.sqrt(3 / 5 + 1 / r5), -1 / 2 * np.sqrt(3 / 5 + 1 / r5),
                 (-1 + r5) / 2 / r10, r2 / r5]
        n2_5d = [-1 / 2 * np.sqrt(1 + 1 / r5), -1 / 2 * np.sqrt(1 - 1 / r5), 1 / 2 * np.sqrt(1 - 1 / r5),
                 1 / 2 * np.sqrt(1 + 1 / r5), 0]
        n3_5d = [1 / r5, 1 / r5, 1 / r5, 1 / r5, 1 / r5]
    basis_5d = [u_5d, v_5d, n1_5d, n2_5d, n3_5d]
    for i, name in enumerate(basis_labels):
        basis_vec = basis_5d[i // 2]
        if i % 2 == 0:
            value = Vector(basis_vec[0:3])
        else:
            value = Vector([*basis_vec[3:5], 0])
        input_vec = InputVector(
            node_tree,
            location=(tl, -1.5 - 0.25 * i),
            value=value,
            name=name,
            label=name,
            hide=True
        )
        input_vecs.append(input_vec)

    tl += 1

    # create 50 linear maps to rotate the basis vectors
    rot_labels = ["rotU", "rotV", "rotN1", "rotN2", "rotN3"]
    input_vec_sockets = []
    for i in range(0, 10, 2):
        last_vec_sockets = [input_vecs[i].std_out, input_vecs[i + 1].std_out]
        for j in range(10):
            lm = LinearMap(node_tree, location=(tl + j, th - 0.4 * i), dimension=5,
                           name="LinearMap" + str(i) + str(9 - j), label=rot_labels[i // 2] + "_" + str(9 - j),
                           hide=True)
            for k in range(10):
                node_tree.links.new(rotations[9 - j].outputs[k], lm.inputs[k + 1])
            node_tree.links.new(last_vec_sockets[0], lm.inputs[11])
            node_tree.links.new(last_vec_sockets[1], lm.inputs[12])
            last_vec_sockets = [lm.outputs[0], lm.outputs[1]]
        input_vec_sockets += last_vec_sockets

    # create sigmas
    s1 = InputVector(node_tree, location=(-5 + length, -4.25),
                     value=Vector([0.2] * 3),
                     name='S1', label='S1', hide=True)
    s2 = InputVector(node_tree, location=(-5 + length, -4.5),
                     value=Vector([0.2, 0.2, 0]),
                     name='S2', label='S2', hide=True)

    sigmas = [s1, s2]
    sigma_labels = ['S1', 'S2']
    length += 1

    two_n_plus_one = nodes.new(type='ShaderNodeMath')
    two_n_plus_one.operation = "MULTIPLY_ADD"
    two_n_plus_one.location = (left + length * width, -150)
    two_n_plus_one.inputs[1].default_value = 2  # Multiplier
    two_n_plus_one.inputs[2].default_value = 1  # Addend
    links.new(n_value.outputs['Value'], two_n_plus_one.inputs['Value'])
    length += 1

    power = nodes.new(type='ShaderNodeMath')
    power.operation = "POWER"
    power.location = (left + length * width, 0)
    power.inputs[1].default_value = 5  # Exponent
    links.new(two_n_plus_one.outputs['Value'], power.inputs[0])
    length += 1

    line = []

    points = Points(node_tree, location=(-5 + length, 1.5), radius=0.01,
                    count=power.outputs['Value'])
    line.append(points)

    index = Index(node_tree, location=(-5 + (length - 1), -1.5))

    index2tuple = make_function(nodes, functions={
        "P1": ['index,base,4,**,/,floor,range,-', 'index,base,4,**,%,base,3,**,/,floor,range,-',
               'index,base,3,**,%,base,2,**,/,floor,range,-'],
        "P2": ['index,base,2,**,%,base,/,floor,range,-', 'index,base,%,range,-', '0']
    }, name="Index2Tuple", inputs=["index", "base", "range"], outputs=["P1", "P2"], scalars=["index", "base", "range"],
                                vectors=["P1", "P2"])
    index2tuple.location = (left + length * width, -300)
    links.new(index.outputs[0], index2tuple.inputs['index'])
    links.new(two_n_plus_one.outputs[0], index2tuple.inputs['base'])
    links.new(n_value.outputs[0], index2tuple.inputs['range'])

    length += 1

    # project to the orthogonal 3 space
    ortho_projection_pp = make_function(node_tree.nodes, functions={
        'Vector': ['n1,P1,S1,sub,dot,m1,P2,S2,sub,dot,+', 'n2,P1,S1,sub,dot,m2,P2,S2,sub,dot,+',
                   'n3,P1,S1,sub,dot,m3,P2,S2,sub,dot,+']
    }, name='OrthgonalProjector',
                                        inputs=ins + sigma_labels + ["P1", "P2"], outputs=['Vector'],
                                        vectors=ins + ["P1", "P2"] + ['Vector'] + sigma_labels)
    ortho_projection_pp.location = (left + length * 200, 0)
    ortho_projection_pp.hide = True

    for i in range(4, 10):
        node_tree.links.new(input_vec_sockets[i], ortho_projection_pp.inputs[basis_labels[i]])
    for name in ["P1", "P2"]:
        node_tree.links.new(index2tuple.outputs[name], ortho_projection_pp.inputs[name])
    for node, label in zip(sigmas, sigma_labels):
        node_tree.links.new(node.std_out, ortho_projection_pp.inputs[label])
    # project to tiling plane

    para_ins = basis_labels[0:4]
    para_projection_pp = make_function(node_tree.nodes, functions={
        'Vector': ['u,P1,S1,sub,dot,U,P2,S2,sub,dot,+', 'v,P1,S1,sub,dot,V,P2,S2,sub,dot,+', '0']
    }, name='ParaProjector', inputs=para_ins + sigma_labels + ["P1", "P2"], outputs=['Vector'],
                                       vectors=para_ins + sigma_labels + ["P1", "P2", "Vector"])
    para_projection_pp.location = (left + length * 200, -50)
    para_projection_pp.hide = True

    for i in range(0, 4):
        node_tree.links.new(input_vec_sockets[i], para_projection_pp.inputs[basis_labels[i]])
    for name in ["P1", "P2"]:
        node_tree.links.new(index2tuple.outputs[name], para_projection_pp.inputs[name])
    for node, label in zip(sigmas, sigma_labels):
        node_tree.links.new(node.std_out, para_projection_pp.inputs[label])

    length += 1

    # create attributes
    for i, name in enumerate(["P1", "P2"]):
        attr = StoredNamedAttribute(node_tree, location=(-5 + length, 1.5),
                                    data_type='FLOAT_VECTOR',
                                    domain='POINT', name=name,
                                    value=index2tuple.outputs[name])
        line.append(attr)
        length += 1

    ico_sphere = IcoSphere(node_tree, location=(-5 + length, 2),
                           radius=0.025, subdivisions=2, hide=True)

    set_pos = SetPosition(node_tree, location=(-5 + length, 1.5),
                          position=ortho_projection_pp.outputs['Vector'],
                          label="Projection3D"
                          )
    line.append(set_pos)
    length += 1

    del_geo_3d = DeleteGeometry(node_tree, location=(-5 + length, 2))
    line.append(del_geo_3d)
    length += 1

    ico_instance = InstanceOnPoints(node_tree, location=(-5 + length, 1.5), instance=ico_sphere.geometry_out,
                                    hide=True)
    line.append(ico_instance)

    # 2D line
    line2 = []
    line2.append(points)

    set_pos2 = SetPosition(node_tree, location=(-5 + length, 0.5),
                           position=para_projection_pp.outputs['Vector'],
                           label="Projection2D")

    line2.append(set_pos2)
    length += 1

    stored_index = StoredNamedAttribute(node_tree, location=(-5 + length, 0.5),
                                        name="SavedIndex",
                                        data_type='INT',
                                        value=index.std_out)
    line2.append(stored_index)
    length += 1

    # create 32 points the form the convex hull with shift possibility
    nl = 0
    nh = -3

    shift1 = InputVector(node_tree, location=(nl - 1, nh))
    shift2 = InputVector(node_tree, location=(nl - 1, nh - 0.25))
    points32 = Points(node_tree, location=(nl, nh), count=32)
    idx = Index(node_tree, location=(nl - 1, nh - 1))

    # this function works also for indices with an arbitrary offset. We don't have to worry,\
    # when the index doesn't start at 0
    mod = make_function(node_tree.nodes, functions={
        'p1': ['index,32,%,16,/,floor,0.,-,shift1_x,+', 'index,16,%,8,/,floor,0.,-,shift1_y,+',
               'index,8,%,4,/,floor,0.,-,shift1_z,+'],
        'p2': ['index,4,%,2,/,floor,0.,-,shift2_x,+', 'index,2,%,0.,-,shift2_y,+', '0']
    }, name="Mod32", inputs=['index', 'shift1', 'shift2'], outputs=['p1', 'p2'], scalars=['index'],
                        vectors=['p1', 'p2', 'shift1', 'shift2'])
    node_tree.links.new(idx.std_out, mod.inputs['index'])
    node_tree.links.new(shift1.std_out, mod.inputs['shift1'])
    node_tree.links.new(shift2.std_out, mod.inputs['shift2'])
    mod.location = (nl * 200, nh * 200 - 200)
    mod.hide = True
    nl += 1

    nline = [points32]
    # create attributes
    for name in ['p1', 'p2']:
        attr = StoredNamedAttribute(node_tree, location=(nl, nh),
                                    data_type='FLOAT_VECTOR', domain='POINT',
                                    name=name, value=mod.outputs[name])
        nline.append(attr)
        nl += 1

    projection32 = make_function(node_tree.nodes, functions={
        'Vector': ['n1,p1,dot,m1,p2,dot,+', 'n2,p1,dot,m2,p2,dot,+', 'n3,p1,dot,m3,p2,dot,+']
    }, name='OrthgonalProjector',
                                 inputs=ins + ["p1", "p2"], outputs=['Vector'], vectors=ins + ["p1", "p2"] + ['Vector'])
    projection32.location = (nl * 200, (nh - 1) * 200)
    projection32.hide = True

    for i in range(4, 10):
        node_tree.links.new(input_vec_sockets[i], projection32.inputs[basis_labels[i]])
    for name in ["p1", "p2"]:
        node_tree.links.new(mod.outputs[name], projection32.inputs[name])

    set_pos32 = SetPosition(node_tree, location=(nl, nh),
                            position=projection32.outputs['Vector']
                            )
    nline.append(set_pos32)
    nl += 1
    hull = ConvexHull(node_tree, location=(nl, nh))
    nline += [hull, WireFrame(node_tree, location=(nl + 1, nh)),
              SetMaterial(node_tree, location=(nl + 2, nh), material='plastic_joker')]
    nl += 2

    # convex hull test with two rays

    convex_hull_test = InsideConvexHull(node_tree, location=(1, -1.25),
                                        target_geometry=hull.geometry_out,
                                        source_position=ortho_projection_pp.outputs['Vector']
                                        )

    node_tree.links.new(convex_hull_test.std_out, set_pos2.node.inputs['Selection'])
    node_tree.links.new(convex_hull_test.std_out, set_pos.node.inputs['Selection'])

    del_geo = DeleteGeometry(node_tree, location=(-5 + length, 0),
                             selection=convex_hull_test.outputs['Is Outside'])
    links.new(convex_hull_test.outputs['Is Outside'], del_geo_3d.inputs['Selection'])

    line2.append(del_geo)

    # create faces in all ten orientations for each selected point
    # and pick only those that completely lie inside the convex hull
    # the face directions are stored in two components of an input vector

    ml = 2
    mh = -0.5

    input_dirs = []
    for i, dir in enumerate(dirs):
        input_dirs.append(InputVector(node_tree, location=(ml, mh - 0.25 * i),
                                      value=dir + [0],
                                      name=str(dir), label=str(dir), hide=True)
                          )

    ml += 1
    mh -= 1

    sel_index = NamedAttribute(node_tree, location=(ml, mh),
                               data_type='INT',
                               name="SavedIndex")

    ml += 1

    sel_index2tuple = make_function(nodes, functions={
        "P1": ['index,base,4,**,/,floor,range,-', 'index,base,4,**,%,base,3,**,/,floor,range,-',
               'index,base,3,**,%,base,2,**,/,floor,range,-'],
        "P2": ['index,base,2,**,%,base,/,floor,range,-', 'index,base,%,range,-', '0']
    }, name="SelectedIndex2Tuple", inputs=["index", "base", "range"], outputs=["P1", "P2"],
                                    scalars=["index", "base", "range"],
                                    vectors=["P1", "P2"])
    sel_index2tuple.location = (ml * width, mh * width)
    links.new(sel_index.std_out, sel_index2tuple.inputs['index'])
    links.new(two_n_plus_one.outputs[0], sel_index2tuple.inputs['base'])
    links.new(n_value.outputs[0], sel_index2tuple.inputs['range'])

    ml += 1

    line3 = []
    sub_join = JoinGeometry(node_tree, location=(nl + 4, 0))
    scale_elements = ScaleElements(node_tree, location=(nl + 5, 0),
                                   scale=0.95)

    if 'final_translation' in kwargs:
        translation = kwargs.pop('final_translation')
    else:
        translation = Vector()
    if 'final_rotation' in kwargs:
        rotation = kwargs.pop('final_rotation')
    else:
        rotation = Vector()
    if 'final_scale' in kwargs:
        scale = kwargs.pop('final_scale')
    else:
        scale = Vector([1, 1, 1])
    transform = TransformGeometry(node_tree, location=(nl + 6, 0),
                                  translation=translation, rotation=rotation, scale=scale)

    line3.append(sub_join)
    line3.append(scale_elements)
    line3.append(transform)
    join = JoinGeometry(node_tree, location=(nl + 7, 0))
    nline.append(join)
    line.append(join)
    # line2.append(join)
    line3.append(join)
    create_geometry_line(node_tree, nline)
    create_geometry_line(node_tree, line)
    create_geometry_line(node_tree, line2)
    create_geometry_line(node_tree, line3)
    create_geometry_line(node_tree, [join_marker, join])
    links.new(join.geometry_out, group_outputs.inputs['Geometry'])

    # make face generation groups

    ml += 1
    for i in range(10):
        direction = str(dirs[i][0] + 1) + str(dirs[i][1] + 1)
        face_generation_group = make_face_generation_group(nodes, basis_labels, sigma_labels,
                                                           material='x' + direction + "_color",
                                                           name="FaceGenerator" + str(dirs[i]))
        face_generation_group.location = ((ml + 0.25 * i) * 200, (mh - 0.25 * i) * 200)
        face_generation_group.hide = True

        node_tree.links.new(del_geo.geometry_out, face_generation_group.inputs['Geometry'])
        node_tree.links.new(input_dirs[i].std_out, face_generation_group.inputs['Direction'])
        node_tree.links.new(sel_index2tuple.outputs["P1"], face_generation_group.inputs['P1'])
        node_tree.links.new(sel_index2tuple.outputs["P2"], face_generation_group.inputs['P2'])
        node_tree.links.new(face_generation_group.outputs["Geometry"], sub_join.inputs['Geometry'])

        for j, label in enumerate(basis_labels):
            node_tree.links.new(input_vec_sockets[j], face_generation_group.inputs[label])
        for node, label in zip(sigmas, sigma_labels):
            node_tree.links.new(node.std_out, face_generation_group.inputs[label])
        node_tree.links.new(hull.geometry_out, face_generation_group.inputs['Convex Hull'])

    return node_tree


def make_face_generation_group(nodes, basis_labels, sigma_labels, material,
                               name="FaceGenerator"):
    width = 200
    length = 0

    group = nodes.new(type='GeometryNodeGroup')
    node_tree = bpy.data.node_groups.new(name, type='GeometryNodeTree')
    group.node_tree = node_tree
    nodes = node_tree.nodes

    group.name = name

    group_inputs = nodes.new('NodeGroupInput')
    group_inputs.location = (0, 0)
    group_outputs = nodes.new('NodeGroupOutput')
    length += 1

    make_new_socket(node_tree, name='Geometry', io='INPUT', type='NodeSocketGeometry')
    make_new_socket(node_tree, name='Direction', io='INPUT', type='NodeSocketVector')
    make_new_socket(node_tree, name='P1', io='INPUT', type='NodeSocketVector')
    make_new_socket(node_tree, name='P2', io='INPUT', type='NodeSocketVector')
    for label in basis_labels:
        make_new_socket(node_tree, name=label, io='INPUT', type='NodeSocketVector')
    for label in sigma_labels:
        make_new_socket(node_tree, name=label, io='INPUT', type='NodeSocketVector')
    make_new_socket(node_tree, name='Convex Hull', io='INPUT', type='NodeSocketGeometry')
    make_new_socket(node_tree, name='Geometry', io='OUTPUT', type='NodeSocketGeometry')

    ins = ["P1", "P2", "dir"] + basis_labels[4:] + sigma_labels
    outs = ["p", "q", "r", "s"]
    # this looks more scary than it is
    # p is just the projection of the point (n1.P, n2.P, n3.P)
    # q is (n1.P+n1.delta_i, n2.P+n2.delta_i,n3.delta_i) which means going one step in i project the resulting point
    # for r one goes one step in i and one in j direciton
    # for q one goes one step in j direction
    orth_face_projector = make_function(nodes, functions={
        'p': ['P1,S1,sub,n1,dot,P2,S2,sub,m1,dot,+', 'P1,S1,sub,n2,dot,P2,S2,sub,m2,dot,+',
              'P1,S1,sub,n3,dot,P2,S2,sub,m3,dot,+'],
        'q': [
            'P1,S1,sub,n1,dot,P2,S2,sub,m1,dot,+,dir_x,0,=,n1_x,*,+,dir_x,1,=,n1_y,*,+,dir_x,2,=,n1_z,*,+,dir_x,3,=,m1_x,*,+,dir_x,4,=,m1_y,*,+',
            'P1,S1,sub,n2,dot,P2,S2,sub,m2,dot,+,dir_x,0,=,n2_x,*,+,dir_x,1,=,n2_y,*,+,dir_x,2,=,n2_z,*,+,dir_x,3,=,m2_x,*,+,dir_x,4,=,m2_y,*,+',
            'P1,S1,sub,n3,dot,P2,S2,sub,m3,dot,+,dir_x,0,=,n3_x,*,+,dir_x,1,=,n3_y,*,+,dir_x,2,=,n3_z,*,+,dir_x,3,=,m3_x,*,+,dir_x,4,=,m3_y,*,+'],
        'r': [
            'P1,S1,sub,n1,dot,P2,S2,sub,m1,dot,+,dir_x,0,=,n1_x,*,+,dir_x,1,=,n1_y,*,+,dir_x,2,=,n1_z,*,+,dir_x,3,=,m1_x,*,+,dir_x,4,=,m1_y,*,+,dir_y,0,=,n1_x,*,+,dir_y,1,=,n1_y,*,+,dir_y,2,=,n1_z,*,+,dir_y,3,=,m1_x,*,+,dir_y,4,=,m1_y,*,+',
            'P1,S1,sub,n2,dot,P2,S2,sub,m2,dot,+,dir_x,0,=,n2_x,*,+,dir_x,1,=,n2_y,*,+,dir_x,2,=,n2_z,*,+,dir_x,3,=,m2_x,*,+,dir_x,4,=,m2_y,*,+,dir_y,0,=,n2_x,*,+,dir_y,1,=,n2_y,*,+,dir_y,2,=,n2_z,*,+,dir_y,3,=,m2_x,*,+,dir_y,4,=,m2_y,*,+',
            'P1,S1,sub,n3,dot,P2,S2,sub,m3,dot,+,dir_x,0,=,n3_x,*,+,dir_x,1,=,n3_y,*,+,dir_x,2,=,n3_z,*,+,dir_x,3,=,m3_x,*,+,dir_x,4,=,m3_y,*,+,dir_y,0,=,n3_x,*,+,dir_y,1,=,n3_y,*,+,dir_y,2,=,n3_z,*,+,dir_y,3,=,m3_x,*,+,dir_y,4,=,m3_y,*,+'],
        's': [
            'P1,S1,sub,n1,dot,P2,S2,sub,m1,dot,+,dir_y,0,=,n1_x,*,+,dir_y,1,=,n1_y,*,+,dir_y,2,=,n1_z,*,+,dir_y,3,=,m1_x,*,+,dir_y,4,=,m1_y,*,+',
            'P1,S1,sub,n2,dot,P2,S2,sub,m2,dot,+,dir_y,0,=,n2_x,*,+,dir_y,1,=,n2_y,*,+,dir_y,2,=,n2_z,*,+,dir_y,3,=,m2_x,*,+,dir_y,4,=,m2_y,*,+',
            'P1,S1,sub,n3,dot,P2,S2,sub,m3,dot,+,dir_y,0,=,n3_x,*,+,dir_y,1,=,n3_y,*,+,dir_y,2,=,n3_z,*,+,dir_y,3,=,m3_x,*,+,dir_y,4,=,m3_y,*,+'],
    },
                                        inputs=ins, outputs=outs, vectors=ins + outs, name='FaceProjector3D'
                                        )

    orth_face_projector.location = (length * width, 0)
    for label in ins:
        if label == 'dir':
            node_tree.links.new(group_inputs.outputs['Direction'], orth_face_projector.inputs[label])
        else:
            node_tree.links.new(group_inputs.outputs[label], orth_face_projector.inputs[label])
    orth_face_projector.hide = True

    ins = ["P1", "P2", "dir"] + basis_labels[0:4] + sigma_labels
    outs = ["p", "q", "r", "s"]
    # this looks more scary than it is
    # p is just the projection of the point (u.P, v.P, 0)
    # q is (u.P+u.delta_i, v.P+v.delta_i,0) which means going one step in i and one step in j direction and project the resulting point
    # q is (u.P+u.delta_i+u.delta_j, v.P+v.delta_i+v.delta_j,0) which means going one step in i and one step in j direction and project the resulting point
    para_face_projector = make_function(nodes, functions={
        'p': ['u,P1,S1,sub,dot,U,P2,S2,sub,dot,+', 'v,P1,S1,sub,dot,V,P2,S2,sub,dot,+', '0'],
        'q': [
            'u,P1,S1,sub,dot,U,P2,S2,sub,dot,+,dir_x,0,=,u_x,*,+,dir_x,1,=,u_y,*,+,dir_x,2,=,u_z,*,+,dir_x,3,=,U_x,*,+,dir_x,4,=,U_y,*,+',
            'v,P1,S1,sub,dot,V,P2,S2,sub,dot,+,dir_x,0,=,v_x,*,+,dir_x,1,=,v_y,*,+,dir_x,2,=,v_z,*,+,dir_x,3,=,V_x,*,+,dir_x,4,=,V_y,*,+',
            '0'],
        'r': [
            'u,P1,S1,sub,dot,U,P2,S2,sub,dot,+,dir_x,0,=,u_x,*,+,dir_x,1,=,u_y,*,+,dir_x,2,=,u_z,*,+,dir_x,3,=,U_x,*,+,dir_x,4,=,U_y,*,+,dir_y,0,=,u_x,*,+,dir_y,1,=,u_y,*,+,dir_y,2,=,u_z,*,+,dir_y,3,=,U_x,*,+,dir_y,4,=,U_y,*,+',
            'v,P1,S1,sub,dot,V,P2,S2,sub,dot,+,dir_x,0,=,v_x,*,+,dir_x,1,=,v_y,*,+,dir_x,2,=,v_z,*,+,dir_x,3,=,V_x,*,+,dir_x,4,=,V_y,*,+,dir_y,0,=,v_x,*,+,dir_y,1,=,v_y,*,+,dir_y,2,=,v_z,*,+,dir_y,3,=,V_x,*,+,dir_y,4,=,V_y,*,+',
            '0'],
        's': [
            'u,P1,S1,sub,dot,U,P2,S2,sub,dot,+,dir_y,0,=,u_x,*,+,dir_y,1,=,u_y,*,+,dir_y,2,=,u_z,*,+,dir_y,3,=,U_x,*,+,dir_y,4,=,U_y,*,+',
            'v,P1,S1,sub,dot,V,P2,S2,sub,dot,+,dir_y,0,=,v_x,*,+,dir_y,1,=,v_y,*,+,dir_y,2,=,v_z,*,+,dir_y,3,=,V_x,*,+,dir_y,4,=,V_y,*,+',
            '0'],
    },
                                        inputs=ins, outputs=outs, vectors=ins + outs, name='FaceProjector2D'
                                        )

    para_face_projector.location = (length * width, -width)
    for label in ins:
        if label == 'dir':
            node_tree.links.new(group_inputs.outputs['Direction'], para_face_projector.inputs[label])
        else:
            node_tree.links.new(group_inputs.outputs[label], para_face_projector.inputs[label])
    para_face_projector.hide = True
    length += 1

    # create for convex hull tests
    tests = []
    points = ['p', 'q', 'r', 's']
    for i in range(4):
        convex_hull_test = InsideConvexHull(node_tree, location=(length, 0 + 0.25 * i),
                                            target_geometry=group_inputs.outputs['Convex Hull'],
                                            source_position=orth_face_projector.outputs[points[i]], hide=True)
        tests.append(convex_hull_test)

    length += 1

    # create and links
    and_pq = BooleanMath(node_tree, location=(length, 0),
                         operation="AND",
                         inputs0=tests[0].std_out,
                         inputs1=tests[1].std_out,
                         name="p and q", hide=True)
    and_pqr = BooleanMath(node_tree, location=(length + 0.25, -0.25),
                          operation="AND",
                          inputs0=and_pq.std_out,
                          inputs1=tests[2].std_out,
                          name="p,q and r", hide=True)
    and_pqrs = BooleanMath(node_tree, location=(length + 0.25, -0.25),
                           operation="AND",
                           inputs0=and_pqr.std_out,
                           inputs1=tests[3].std_out,
                           name="p,q and r", hide=True)
    length += 1
    # create geometry line
    points_to_vertices = PointsToVertices(node_tree, location=(length, 1),
                                          selection=and_pqrs.std_out, hide=True)

    q_minus_p = VectorMath(node_tree, location=(length, 0.5),
                           operation="SUBTRACT",
                           inputs0=para_face_projector.outputs["q"],
                           inputs1=para_face_projector.outputs["p"],
                           name="q-p", hide=True)
    length += 1
    make_lines = ExtrudeMesh(node_tree, location=(length, 1),
                             mode='VERTICES',
                             offset=q_minus_p.std_out,
                             name="MakeLines", hide=True)
    length += 1
    s_minus_p = VectorMath(node_tree, location=(length, 0.5),
                           operation="SUBTRACT",
                           inputs0=para_face_projector.outputs["s"],
                           inputs1=para_face_projector.outputs["p"],
                           name="s-p", hide=True)
    make_faces = ExtrudeMesh(node_tree, location=(length, 1),
                             mode='EDGES',
                             offset=s_minus_p.std_out,
                             name="MakeLines", hide=True)
    length += 1
    set_material = SetMaterial(node_tree, location=(length, 1),
                               material=material
                               )
    length += 1

    create_geometry_line(node_tree, [points_to_vertices, make_lines, make_faces, set_material],
                         ins=group_inputs.outputs['Geometry'], out=group_outputs.inputs['Geometry'])

    group_outputs.location = (length * width, 0)
    return group


def make_face_generation_group_3d(nodes, basis_labels, sigma_labels, material,
                                  name="FaceGenerator"):
    width = 200
    length = 0

    group = nodes.new(type='GeometryNodeGroup')
    node_tree = bpy.data.node_groups.new(name, type='GeometryNodeTree')
    group.node_tree = node_tree
    nodes = node_tree.nodes

    group.name = name

    group_inputs = nodes.new('NodeGroupInput')
    group_inputs.location = (0, 0)
    group_outputs = nodes.new('NodeGroupOutput')
    length += 1

    make_new_socket(node_tree, name='Geometry', io='INPUT', type='NodeSocketGeometry')
    make_new_socket(node_tree, name='Direction', io='INPUT', type='NodeSocketVector')
    make_new_socket(node_tree, name='P', io='INPUT', type='NodeSocketVector')
    for label in basis_labels:
        make_new_socket(node_tree, name=label, io='INPUT', type='NodeSocketVector')
    for label in sigma_labels:
        make_new_socket(node_tree, name=label, io='INPUT', type='NodeSocketVector')
    make_new_socket(node_tree, name='Convex Hull', io='INPUT', type='NodeSocketGeometry')
    make_new_socket(node_tree, name='Geometry', io='OUTPUT', type='NodeSocketGeometry')

    ins = ["P", "dir"] + ["n"] + sigma_labels
    outs = ["p", "q", "r", "s"]
    orth_face_projector = make_function(nodes, functions={
        'p': ['P,sigma,sub,n,dot'],
        'q':
            ['P,sigma,sub,n,dot,dir_x,0,=,n_x,*,+,dir_x,1,=,n_y,*,+,dir_x,2,=,n_z,*,+'],
        'r': [
            'P,sigma,sub,n,dot,dir_x,0,=,n_x,*,+,dir_x,1,=,n_y,*,+,dir_x,2,=,n_z,*,+,dir_y,0,=,n_x,*,+,dir_y,1,=,n_y,*,+,dir_y,2,=,n_z,*,+'],
        's': ['P,sigma,sub,n,dot,dir_y,0,=,n_x,*,+,dir_y,1,=,n_y,*,+,dir_y,2,=,n_z,*,+']
    }, inputs=ins, outputs=outs, vectors=ins, scalars=outs, name='FaceProjector3D')

    orth_face_projector.location = (length * width, 0)
    for label in ins:
        if label == 'dir':
            node_tree.links.new(group_inputs.outputs['Direction'], orth_face_projector.inputs[label])
        else:
            node_tree.links.new(group_inputs.outputs[label], orth_face_projector.inputs[label])
    orth_face_projector.hide = True

    ins = ["P", "dir", "u", "v"] + ["n"] + sigma_labels
    outs = ["p", "q", "r", "s"]
    para_face_projector = make_function(nodes, functions={
        'p': ['u,P,sigma,sub,dot', 'v,P,sigma,sub,dot', '0'],
        'q': [
            'u,P,sigma,sub,dot,dir_x,0,=,u_x,*,+,dir_x,1,=,u_y,*,+,dir_x,2,=,u_z,*,+',
            'v,P,sigma,sub,dot,dir_x,0,=,v_x,*,+,dir_x,1,=,v_y,*,+,dir_x,2,=,v_z,*,+',
            '0'],
        'r': [
            'u,P,sigma,sub,dot,dir_x,0,=,u_x,*,+,dir_x,1,=,u_y,*,+,dir_x,2,=,u_z,*,+,dir_y,0,=,u_x,*,+,dir_y,1,=,u_y,*,+,dir_y,2,=,u_z,*,+',
            'v,P,sigma,sub,dot,dir_x,0,=,v_x,*,+,dir_x,1,=,v_y,*,+,dir_x,2,=,v_z,*,+,dir_y,0,=,v_x,*,+,dir_y,1,=,v_y,*,+,dir_y,2,=,v_z,*,+',
            '0'],
        's': [
            'u,P,sigma,sub,dot,dir_y,0,=,u_x,*,+,dir_y,1,=,u_y,*,+,dir_y,2,=,u_z,*,+',
            'v,P,sigma,sub,dot,dir_y,0,=,v_x,*,+,dir_y,1,=,v_y,*,+,dir_y,2,=,v_z,*,+',
            '0'],
    },
                                        inputs=ins, outputs=outs, vectors=ins + outs, name='FaceProjector2D'
                                        )

    para_face_projector.location = (length * width, -width)
    for label in ins:
        if label == 'dir':
            node_tree.links.new(group_inputs.outputs['Direction'], para_face_projector.inputs[label])
        else:
            node_tree.links.new(group_inputs.outputs[label], para_face_projector.inputs[label])
    para_face_projector.hide = True
    length += 1

    # create for convex hull tests
    tests = []
    points = ['p', 'q', 'r', 's']
    for i in range(4):
        convex_hull_test = InsideConvexHull3D(node_tree, location=(length, 0 + 0.25 * i),
                                              target_geometry=group_inputs.outputs['Convex Hull'],
                                              source_position=orth_face_projector.outputs[points[i]], hide=True)
        tests.append(convex_hull_test)

    length += 1

    # create and links
    and_pq = BooleanMath(node_tree, location=(length, 0),
                         operation="AND",
                         inputs0=tests[0].std_out,
                         inputs1=tests[1].std_out,
                         name="p and q", hide=True)
    and_pqr = BooleanMath(node_tree, location=(length + 0.25, -0.25),
                          operation="AND",
                          inputs0=and_pq.std_out,
                          inputs1=tests[2].std_out,
                          name="p,q and r", hide=True)
    and_pqrs = BooleanMath(node_tree, location=(length + 0.25, -0.25),
                           operation="AND",
                           inputs0=and_pqr.std_out,
                           inputs1=tests[3].std_out,
                           name="p,q and r", hide=True)
    length += 1
    # create geometry line
    points_to_vertices = PointsToVertices(node_tree, location=(length, 1),
                                          selection=and_pqrs.std_out, hide=True)

    q_minus_p = VectorMath(node_tree, location=(length, 0.5),
                           operation="SUBTRACT",
                           inputs0=para_face_projector.outputs["q"],
                           inputs1=para_face_projector.outputs["p"],
                           name="q-p", hide=True)
    length += 1
    make_lines = ExtrudeMesh(node_tree, location=(length, 1),
                             mode='VERTICES',
                             offset=q_minus_p.std_out,
                             name="MakeLines", hide=True)
    length += 1
    s_minus_p = VectorMath(node_tree, location=(length, 0.5),
                           operation="SUBTRACT",
                           inputs0=para_face_projector.outputs["s"],
                           inputs1=para_face_projector.outputs["p"],
                           name="s-p", hide=True)
    make_faces = ExtrudeMesh(node_tree, location=(length, 1),
                             mode='EDGES',
                             offset=s_minus_p.std_out,
                             name="MakeLines", hide=True)
    length += 1
    set_material = SetMaterial(node_tree, location=(length, 1),
                               material=material
                               )
    length += 1

    create_geometry_line(node_tree, [points_to_vertices, make_lines, make_faces, set_material],
                         ins=group_inputs.outputs['Geometry'], out=group_outputs.inputs['Geometry'])

    group_outputs.location = (length * width, 0)
    return group


def make_face_generation_group_3d_unprojected(nodes, basis_labels, sigma_labels, material,
                                              name="FaceGenerator"):
    width = 200
    length = 0

    group = nodes.new(type='GeometryNodeGroup')
    node_tree = bpy.data.node_groups.new(name, type='GeometryNodeTree')
    group.node_tree = node_tree
    nodes = node_tree.nodes

    group.name = name

    group_inputs = nodes.new('NodeGroupInput')
    group_inputs.location = (0, 0)
    group_outputs = nodes.new('NodeGroupOutput')
    length += 1

    make_new_socket(node_tree, name='Geometry', io='INPUT', type='NodeSocketGeometry')
    make_new_socket(node_tree, name='Direction', io='INPUT', type='NodeSocketVector')
    make_new_socket(node_tree, name='FaceScale', io='INPUT', type='NodeSocketFloat')
    make_new_socket(node_tree, name='P', io='INPUT', type='NodeSocketVector')
    for label in basis_labels:
        make_new_socket(node_tree, name=label, io='INPUT', type='NodeSocketVector')
    for label in sigma_labels:
        make_new_socket(node_tree, name=label, io='INPUT', type='NodeSocketVector')
    make_new_socket(node_tree, name='Convex Hull', io='INPUT', type='NodeSocketGeometry')
    make_new_socket(node_tree, name='Geometry', io='OUTPUT', type='NodeSocketGeometry')

    ins = ["P", "dir"] + ["n"] + sigma_labels
    outs = ["p", "q", "r", "s"]
    orth_face_projector = make_function(nodes, functions={
        'p': ['P,sigma,sub,n,dot'],
        'q':
            ['P,sigma,sub,n,dot,dir_x,0,=,n_x,*,+,dir_x,1,=,n_y,*,+,dir_x,2,=,n_z,*,+'],
        'r': [
            'P,sigma,sub,n,dot,dir_x,0,=,n_x,*,+,dir_x,1,=,n_y,*,+,dir_x,2,=,n_z,*,+,dir_y,0,=,n_x,*,+,dir_y,1,=,n_y,*,+,dir_y,2,=,n_z,*,+'],
        's': ['P,sigma,sub,n,dot,dir_y,0,=,n_x,*,+,dir_y,1,=,n_y,*,+,dir_y,2,=,n_z,*,+']
    }, inputs=ins, outputs=outs, vectors=ins, scalars=outs, name='FaceProjector3D')

    orth_face_projector.location = (length * width, 0)
    for label in ins:
        if label == 'dir':
            node_tree.links.new(group_inputs.outputs['Direction'], orth_face_projector.inputs[label])
        else:
            node_tree.links.new(group_inputs.outputs[label], orth_face_projector.inputs[label])
    orth_face_projector.hide = True
    length += 1

    # create for convex hull tests
    tests = []
    points = ['p', 'q', 'r', 's']
    for i in range(4):
        convex_hull_test = InsideConvexHull3D(node_tree, location=(length, 0 + 0.25 * i),
                                              target_geometry=group_inputs.outputs['Convex Hull'],
                                              source_position=orth_face_projector.outputs[points[i]], hide=True)
        tests.append(convex_hull_test)

    length += 1

    # create and links
    and_pq = BooleanMath(node_tree, location=(length, 0),
                         operation="AND",
                         inputs0=tests[0].std_out,
                         inputs1=tests[1].std_out,
                         name="p and q", hide=True)
    and_pqr = BooleanMath(node_tree, location=(length + 0.25, -0.25),
                          operation="AND",
                          inputs0=and_pq.std_out,
                          inputs1=tests[2].std_out,
                          name="p,q and r", hide=True)
    and_pqrs = BooleanMath(node_tree, location=(length + 0.25, -0.25),
                           operation="AND",
                           inputs0=and_pqr.std_out,
                           inputs1=tests[3].std_out,
                           name="p,q and r", hide=True)
    length += 1
    # create geometry line
    points_to_vertices = PointsToVertices(node_tree, location=(length, 1),
                                          selection=and_pqrs.std_out, hide=True)

    length += 1
    make_lines = ExtrudeMesh(node_tree, location=(length, 1),
                             mode='VERTICES',
                             offset=group_inputs.outputs['u'],
                             name="MakeLines", hide=True)
    length += 1
    make_faces = ExtrudeMesh(node_tree, location=(length, 1),
                             mode='EDGES',
                             offset=group_inputs.outputs['v'],
                             name="MakeLines", hide=True)
    length += 1
    set_material = SetMaterial(node_tree, location=(length, 1),
                               material=material, emission=0.0
                               )
    length += 1

    scale_element = ScaleElements(node_tree, location=(length, 1),
                                  scale=group_inputs.outputs['FaceScale'],
                                  )
    length += 1

    create_geometry_line(node_tree, [points_to_vertices, make_lines, make_faces, scale_element, set_material],
                         ins=group_inputs.outputs['Geometry'], out=group_outputs.inputs['Geometry'])

    group_outputs.location = (length * width, 0)
    return group


def create_index2tuple_function(nodes, scalar_input_parameters=['index', 'base', 'range'],
                                dim=5, name='Index2Tuple',
                                node_tree_type='Geometry'):
    """
    converts an index into a tuple of dimension dim
    with integer components in the range: -range ... range

    :param nodes:
    :param scalar_input_parameters: range=3, base =2*3+1, index = base**dim
    :param dim:
    :param name:
    :param node_tree_type:
    :return:
    """
    i_label = scalar_input_parameters[0]
    b_label = scalar_input_parameters[1]
    r_label = scalar_input_parameters[2]

    if node_tree_type == 'Shader':
        tree = bpy.data.node_groups.new(type='ShaderNodeTree', name=name)
        group = nodes.new(type='ShaderNodeGroup')
    else:
        tree = bpy.data.node_groups.new(type='GeometryNodeTree', name=name)
        group = nodes.new(type='GeometryNodeGroup')

    tree_nodes = tree.nodes
    tree_links = tree.links

    width = 200
    length = -1

    group_inputs = tree_nodes.new('NodeGroupInput')
    group_outputs = tree_nodes.new('NodeGroupOutput')

    for param in scalar_input_parameters:
        make_new_socket(tree, name=param, io='INPUT', type='NodeSocketFloat')

    for i in range(dim):
        make_new_socket(tree, name='d' + str(i + 1), io='OUTPUT', type='NodeSocketFloat')

    group.name = name
    group.node_tree = tree
    group_inputs.location = (length * width, 0)
    length = 0
    i_socket = group_inputs.outputs[i_label]

    power = tree_nodes.new(type='ShaderNodeMath')
    power.operation = "POWER"
    power.location = (length * width, 200)
    power.inputs[1].default_value = dim - 1
    tree_links.new(group_inputs.outputs[b_label], power.inputs[0])
    length += 1
    b_socket = power.outputs['Value']

    range_socket = group_inputs.outputs[r_label]

    for i in range(dim):
        height = 1000 - 3 * (i + 1) * 200
        delta = tree_nodes.new(type='ShaderNodeMath')
        delta.operation = "DIVIDE"
        delta.location = (length * width, height)
        tree_links.new(i_socket, delta.inputs[0])
        tree_links.new(b_socket, delta.inputs[1])

        floor = tree_nodes.new(type='ShaderNodeMath')
        floor.operation = "FLOOR"
        floor.location = ((length + 1) * width, height)
        tree_links.new(delta.outputs[0], floor.inputs[0])

        result = tree_nodes.new(type='ShaderNodeMath')
        result.operation = "SUBTRACT"
        result.location = ((length + 2) * width, height)
        tree_links.new(range_socket, result.inputs[1])
        tree_links.new(floor.outputs[0], result.inputs[0])
        tree_links.new(result.outputs[0], group_outputs.inputs['d' + str(i + 1)])

        tmp = tree_nodes.new(type='ShaderNodeMath')
        tmp.operation = "MULTIPLY"
        tmp.location = (length * width, height - 200)
        tree_links.new(floor.outputs[0], tmp.inputs[0])
        tree_links.new(b_socket, tmp.inputs[1])

        i_new = tree_nodes.new(type='ShaderNodeMath')
        i_new.operation = "SUBTRACT"
        i_new.location = ((length + 1) * width, height - 200)
        tree_links.new(i_socket, i_new.inputs[0])
        tree_links.new(tmp.outputs[0], i_new.inputs[1])

        b_new = tree_nodes.new(type='ShaderNodeMath')
        b_new.operation = "DIVIDE"
        b_new.location = (length * width, height - 400)
        tree_links.new(b_socket, b_new.inputs[0])
        tree_links.new(group_inputs.outputs[b_label], b_new.inputs[1])

        i_socket = i_new.outputs[0]
        b_socket = b_new.outputs[0]
    length += 3
    group_outputs.location = (length * width, 0)

    return group


###########################
## Penrose Tiling De Bruijn unpublished
###########################

def de_bruijn(name='DeBruijnNode', k=3, tile_separation=0.05, base_color='drawing', **kwargs):
    """
       create a geometry node that computes the Penrose tiling from penta grids

       :param base_color:
       :param tile_separation: 0 means the tiles are displayed at 100% touching each other completely
                               0.1 means that the tiles are only displayed at 90% of their size
       :param name:
       :param k: each grid ranges from -k,...,k
       :return:
       """

    node_tree = bpy.data.node_groups.new(name, type='GeometryNodeTree')
    nodes = node_tree.nodes
    links = node_tree.links

    left = -1000
    width = 200
    length = 0

    group_outputs = nodes.new('NodeGroupOutput')
    length += 1

    make_new_socket(node_tree, name='Geometry', io='OUTPUT', type='NodeSocketGeometry')

    # create point cloud
    k_value = nodes.new(type="ShaderNodeValue")
    k_value.name = 'range'
    k_value.label = 'range'
    k_value.outputs[0].default_value = k
    k_value.location = (left + length * width, -150)
    k_value.hide = True
    length += 1

    two_k_plus_one = nodes.new(type='ShaderNodeMath')
    two_k_plus_one.operation = "MULTIPLY_ADD"
    two_k_plus_one.location = (left + length * width, -150)
    two_k_plus_one.inputs[1].default_value = 2  # Multiplier
    two_k_plus_one.inputs[2].default_value = 1  # Addend
    links.new(k_value.outputs['Value'], two_k_plus_one.inputs['Value'])
    length += 1

    # create (2k+1)**2*10 points

    power = nodes.new(type='ShaderNodeMath')
    power.operation = "POWER"
    power.location = (left + length * width, 0)
    power.inputs[1].default_value = 2  # Exponent
    links.new(two_k_plus_one.outputs['Value'], power.inputs[0])
    length += 1

    num_points = nodes.new(type='ShaderNodeMath')
    num_points.operation = "MULTIPLY"
    num_points.location = (left + length * width, 0)
    num_points.inputs[1].default_value = 10
    links.new(power.outputs['Value'], num_points.inputs[0])
    length += 1

    points = nodes.new(type='GeometryNodePoints')
    points.location = (left + length * width, 300)
    points.inputs['Radius'].default_value = 0.01
    links.new(num_points.outputs['Value'], points.inputs['Count'])
    geometry_line = length + 4

    index = nodes.new(type="GeometryNodeInputIndex")
    index.location = (left + (length - 1) * width, -300)

    index2ijlm = create_index2ijlm_function(nodes, scalar_input_parameters=['index', 'base', 'range'],
                                            name='Index2ijlm')
    index2ijlm.location = (left + length * width, -300)
    links.new(index.outputs[0], index2ijlm.inputs['index'])
    links.new(two_k_plus_one.outputs[0], index2ijlm.inputs['base'])
    links.new(k_value.outputs[0], index2ijlm.inputs['range'])

    # create basis vectors

    us = []
    for i in range(5):
        basis = nodes.new(type="FunctionNodeInputVector")
        us.append(basis)
        basis.name = 'u' + str(i)
        basis.label = 'u' + str(i)
        basis.vector = [np.cos(2 * np.pi / 5 * i), np.sin(2 * np.pi / 5 * i), 0]
        basis.location = (left + length * width - 200, -550 - 50 * i)
        basis.hide = True

    vs = []
    for i in range(5):
        basis = nodes.new(type="FunctionNodeInputVector")
        vs.append(basis)
        basis.name = 'v' + str(i)
        basis.label = 'v' + str(i)
        basis.vector = [-np.sin(2 * np.pi / 5 * i), np.cos(2 * np.pi / 5 * i), 0]
        basis.location = (left + length * width, -550 - 50 * i)
        basis.hide = True

    # create sigmas
    sigmas = []
    for i in range(4):
        sigma = nodes.new(type="ShaderNodeValue")
        sigmas.append(sigma)
        sigma.name = 's' + str(i)
        sigma.label = 's' + str(i)
        if i < 4:
            sigma.outputs[0].default_value = 0.2
        else:
            sigma.outputs[0].default_value = -0.8
        sigma.location = (left + length * width, -800 - 50 * i)
        sigma.hide = True

    s4 = make_function(nodes, functions={
        's4': '0,s0,-,s1,-,s2,-,s3,-'
    }, name='s4', inputs=['s0', 's1', 's2', 's3'], outputs=['s4'], scalars=['s' + str(i) for i in range(5)])
    s4.location = (left + (length + 0.5) * width, -1000)
    s4.hide = True
    sigmas.append(s4)

    for i in range(4):
        links.new(sigmas[i].outputs[0], s4.inputs[i])
    length += 1

    pick_vi = pick_vector_by_index(nodes, array_name='v', range=list(range(5)), name='vi')
    pick_vi.location = (left + length * width, -100)
    pick_vi.hide = True
    links.new(index2ijlm.outputs['i'], pick_vi.inputs['In'])
    for i in range(5):
        links.new(vs[i].outputs[0], pick_vi.inputs['v' + str(i)])

    pick_vj = pick_vector_by_index(nodes, array_name='v', range=list(range(5)), name='vj')
    pick_vj.location = (left + length * width, -150)
    pick_vj.hide = True
    links.new(index2ijlm.outputs['j'], pick_vj.inputs['In'])
    for i in range(5):
        links.new(vs[i].outputs[0], pick_vj.inputs['v' + str(i)])

    pick_ui = pick_vector_by_index(nodes, array_name='u', range=list(range(5)), name='ui')
    pick_ui.location = (left + length * width, -200)
    pick_ui.hide = True
    links.new(index2ijlm.outputs['i'], pick_ui.inputs['In'])
    for i in range(5):
        links.new(us[i].outputs[0], pick_ui.inputs['u' + str(i)])

    pick_uj = pick_vector_by_index(nodes, array_name='u', range=list(range(5)), name='uj')
    pick_uj.location = (left + length * width, -250)
    pick_uj.hide = True
    links.new(index2ijlm.outputs['j'], pick_uj.inputs['In'])
    for i in range(5):
        links.new(us[i].outputs[0], pick_uj.inputs['u' + str(i)])
    length += 1

    intersections = create_intersection_calculator(nodes)
    intersections.location = (left + length * width, -300)
    length += 1

    links.new(pick_vi.outputs[0], intersections.inputs['vi'])
    links.new(pick_vj.outputs[0], intersections.inputs['vj'])

    for i in range(5):
        links.new(sigmas[i].outputs[0], intersections.inputs['s' + str(i)])
    for idx in ['i', 'j', 'l', 'm']:
        links.new(index2ijlm.outputs[idx], intersections.inputs[idx])

    # lift intersections to 5d
    in_scalars = ['s' + str(i) for i in range(5)]
    vectors = ['v' + str(i) for i in range(5)]
    out_scalars = ['I', 'J', 'K', 'L', 'M']
    lift_5D = make_function(nodes,
                            functions={
                                'I': "v0,intersection,dot,s0,-,ceil",
                                'J': "v1,intersection,dot,s1,-,ceil",
                                'K': "v2,intersection,dot,s2,-,ceil",
                                'L': "v3,intersection,dot,s3,-,ceil",
                                'M': "v4,intersection,dot,s4,-,ceil",
                            },
                            inputs=['intersection'] + vectors + in_scalars,
                            outputs=out_scalars,
                            vectors=['intersection'] + vectors,
                            scalars=in_scalars + out_scalars,
                            name='Lift5D'
                            )

    lift_5D.location = (left + length * width, -300)
    for i, label in enumerate(in_scalars):
        links.new(sigmas[i].outputs[0], lift_5D.inputs[label])
    for i, label in enumerate(vectors):
        links.new(vs[i].outputs[0], lift_5D.inputs[label])
    links.new(intersections.outputs[0], lift_5D.inputs['intersection'])
    length += 1

    re_index = make_function(nodes, functions={
        'I': "i,0,=,l,*,i,0,>,I,*,+",
        'J': "i,1,=,l,*,j,1,=,m,*,+,i,1,-,abs,0,>,j,1,-,abs,0,>,*,J,*,+",
        'K': "i,2,=,l,*,j,2,=,m,*,+,i,2,-,abs,0,>,j,2,-,abs,0,>,*,K,*,+",
        'L': "i,3,=,l,*,j,3,=,m,*,+,i,3,-,abs,0,>,j,3,-,abs,0,>,*,L,*,+",
        'M': "i,4,=,l,*,j,4,=,m,*,+,i,4,-,abs,0,>,j,4,-,abs,0,>,*,M,*,+",
    }, name='ReIndexing', inputs=['i', 'j', 'l', 'm', 'I', 'J', 'K', 'L', 'M'], outputs=out_scalars,
                             scalars=['i', 'j', 'l', 'm', 'I', 'J', 'K', 'L', 'M'])
    re_index.location = (left + length * width, -150)
    length += 1

    for name in out_scalars:
        links.new(lift_5D.outputs[name], re_index.inputs[name])
    for name in ['i', 'j', 'l', 'm']:
        links.new(index2ijlm.outputs[name], re_index.inputs[name])

    idx_5D = ['I', 'J', 'K', 'L', 'M']
    vectors = ['v' + str(i) for i in range(5)]
    scalars = ['s' + str(i) for i in range(5)]
    penrose_point = make_function(nodes, functions={
        'pen': 'v0,I,s0,+,scale,v1,J,s1,+,scale,add,v2,K,s2,+,scale,add,v3,L,s3,+,scale,add,v4,M,s4,+,scale,add'
    }, name='PenrosePoint', inputs=idx_5D + vectors + scalars, outputs=['pen'], scalars=idx_5D + scalars,
                                  vectors=['pen'] + vectors)
    penrose_point.location = (left + length * width, -150)
    length += 1

    for name in idx_5D:
        links.new(re_index.outputs[name], penrose_point.inputs[name])
    for i, v in enumerate(vectors):
        links.new(vs[i].outputs[0], penrose_point.inputs[v])
    for i, s in enumerate(scalars):
        links.new(sigmas[i].outputs[0], penrose_point.inputs[s])

    links.new(penrose_point.outputs[0], points.inputs['Position'])

    corners = make_function(nodes, functions={
        'p0': 'p',
        'p1': 'p,vi,add',
        'p2': 'p,vi,add,vj,add',
        'p3': 'p,vj,add'
    }, name='Corners', inputs=['p', 'vi', 'vj'], outputs=['p' + str(i) for i in range(4)],
                            vectors=['p', 'vi', 'vj'] + ['p' + str(i) for i in range(4)]
                            )
    corners.location = (left + length * width, -150)
    length += 1

    links.new(penrose_point.outputs[0], corners.inputs['p'])
    links.new(pick_vi.outputs[0], corners.inputs['vi'])
    links.new(pick_vj.outputs[0], corners.inputs['vj'])

    # stretcher

    stretcher = nodes.new(type="ShaderNodeVectorMath")
    stretcher.operation = 'SCALE'
    stretcher.label = 'stretcher'
    stretcher.name = 'stretcher'
    stretcher.hide = True
    stretcher.location = (left + geometry_line * width, 25)
    links.new(corners.outputs['p0'], stretcher.inputs[0])
    links.new(stretcher.outputs[0], points.inputs['Position'])
    # geometry line

    points2mesh = nodes.new(type="GeometryNodePointsToVertices")
    points2mesh.location = (left + (geometry_line * width), 250)
    geometry_line += 1
    links.new(points.outputs['Geometry'], points2mesh.inputs['Points'])

    diff = nodes.new(type="ShaderNodeVectorMath")
    diff.operation = 'SUBTRACT'
    diff.label = 'p1-p0'
    diff.hide = True
    diff.location = (left + geometry_line * width, 75)
    links.new(corners.outputs['p1'], diff.inputs[0])
    links.new(corners.outputs['p0'], diff.inputs[1])
    geometry_line += 1

    extrude2line = nodes.new(type="GeometryNodeExtrudeMesh")
    extrude2line.location = (left + (geometry_line * width), 200)
    extrude2line.mode = 'VERTICES'
    geometry_line += 1
    links.new(points2mesh.outputs['Mesh'], extrude2line.inputs['Mesh'])
    links.new(diff.outputs[0], extrude2line.inputs['Offset'])

    diff2 = nodes.new(type="ShaderNodeVectorMath")
    diff2.operation = 'SUBTRACT'
    diff2.label = 'p2-p1'
    diff2.hide = True
    diff2.location = (left + geometry_line * width, 25)
    links.new(corners.outputs['p2'], diff2.inputs[0])
    links.new(corners.outputs['p1'], diff2.inputs[1])
    geometry_line += 1

    extrude2face = nodes.new(type="GeometryNodeExtrudeMesh")
    extrude2face.location = (left + geometry_line * width, 150)
    extrude2face.mode = 'EDGES'
    geometry_line += 1
    links.new(extrude2line.outputs['Mesh'], extrude2face.inputs['Mesh'])
    links.new(diff2.outputs[0], extrude2face.inputs['Offset'])

    scale = nodes.new(type='GeometryNodeScaleElements')
    scale.location = (left + geometry_line * width, 100)
    scale.inputs['Scale'].default_value = 1 - tile_separation
    scale.name = 'TileSize'
    scale.label = 'tile_size'
    geometry_line += 1
    links.new(extrude2face.outputs['Mesh'], scale.inputs['Geometry'])

    # index calc
    inside_point = make_function(nodes, functions={
        'inside': 'p0,ui,uj,add,0.001,scale,-'
    }, name='InsidePoint', inputs=['p0', 'ui', 'uj'], outputs=['inside'], vectors=['p0', 'ui', 'uj', 'inside'])
    inside_point.location = (left + width * length, -500)
    links.new(corners.outputs['p0'], inside_point.inputs['p0'])
    links.new(pick_ui.outputs[0], inside_point.inputs['ui'])
    links.new(pick_uj.outputs[0], inside_point.inputs['uj'])

    ins = ['z'] + ['v' + str(i) for i in range(5)] + ['s' + str(i) for i in range(5)]
    index = make_function(nodes, functions={
        'idx': 'z,v0,dot,s0,-,ceil,z,v1,dot,s1,-,ceil,+,z,v2,dot,s2,-,ceil,+,z,v3,dot,s3,-,ceil,+,z,v4,dot,s4,-,ceil,+'
    }, name='Index', inputs=ins, vectors=ins, outputs=['idx'], scalars=['idx'])
    index.location = (left + width * length + 50, -600)

    for i in range(5):
        links.new(vs[i].outputs[0], index.inputs['v' + str(i)])
        links.new(sigmas[i].outputs[0], index.inputs['s' + str(i)])
    links.new(inside_point.outputs[0], index.inputs['z'])

    attr = nodes.new(type="GeometryNodeStoreNamedAttribute")
    attr.location = (left + geometry_line * width, 50)
    attr.data_type = 'INT'
    attr.domain = 'POINT'
    attr.inputs['Name'].default_value = 'Index'
    links.new(scale.outputs['Geometry'], attr.inputs['Geometry'])
    links.new(attr.outputs['Geometry'], group_outputs.inputs['Geometry'])
    links.new(index.outputs['idx'], attr.inputs['Value'])
    geometry_line += 1

    # area attribute
    area = nodes.new(type='GeometryNodeInputMeshFaceArea')
    area.location = (left + (geometry_line - 1) * width, -300)

    attr2 = nodes.new(type="GeometryNodeStoreNamedAttribute")
    attr2.location = (left + geometry_line * width, 50)
    attr2.data_type = 'FLOAT'
    attr2.domain = 'FACE'
    attr2.inputs['Name'].default_value = 'Size'
    links.new(attr.outputs['Geometry'], attr2.inputs['Geometry'])
    links.new(area.outputs['Area'], attr2.inputs['Value'])
    geometry_line += 1

    # center attribute
    ins = ['p' + str(i) for i in range(4)]
    center = make_function(nodes, functions={
        'center': 'p0,p1,add,p2,add,p3,add,0.25,scale'
    }, name="CenterOfTile", inputs=ins, outputs=['center'], vectors=ins + ['center']
                           )
    center.location = (left + (geometry_line - 1) * width, -300)
    center.hide = True
    for i in ins:
        links.new(corners.outputs[i], center.inputs[i])

    attr3 = nodes.new(type="GeometryNodeStoreNamedAttribute")
    attr3.location = (left + geometry_line * width, 50)
    attr3.data_type = 'FLOAT_VECTOR'
    attr3.domain = 'FACE'
    attr3.inputs['Name'].default_value = 'Center'
    links.new(attr2.outputs['Geometry'], attr3.inputs['Geometry'])
    links.new(center.outputs['center'], attr3.inputs['Value'])
    geometry_line += 1

    if 'radius' in kwargs:
        radius = kwargs.pop('radius')
    else:
        radius = None

    if radius:
        delete_geo = nodes.new(type='GeometryNodeDeleteGeometry')
        delete_geo.location = (left + geometry_line * width, 60)
        delete_geo.domain = 'FACE'

        selector = make_function(nodes, functions={
            'isGreater': 'center,length,' + str(radius) + ',>'
        }, name="Selector: l>" + str(radius), inputs=['center'], outputs=['isGreater'], vectors=['center'],
                                 scalars=['isGreater'])
        selector.location = (left + (geometry_line - 1) * width, -300)
        selector.label = selector.name
        selector.hide = True
        links.new(center.outputs['center'], selector.inputs['center'])
        links.new(selector.outputs['isGreater'], delete_geo.inputs['Selection'])
        links.new(attr3.outputs['Geometry'], delete_geo.inputs['Geometry'])
        geometry_line += 1
        last = delete_geo.outputs['Geometry']
    else:
        last = attr3.outputs['Geometry']

    set_material = nodes.new(type='GeometryNodeSetMaterial')
    set_material.location = (left + geometry_line * width, 50)
    set_material.inputs[2].default_value = penrose_material(base_color=base_color, **kwargs)
    geometry_line += 1

    links.new(last, set_material.inputs['Geometry'])

    out = group_outputs.inputs['Geometry']
    group_outputs.location = (left + (geometry_line) * width, 0)
    last = set_material.outputs['Geometry']
    links.new(last, out)

    geometry_line += 1
    return node_tree


def create_intersection_calculator(nodes, name='Intersection'):
    tree = bpy.data.node_groups.new(type='GeometryNodeTree', name='Intersection')
    group = nodes.new(type='GeometryNodeGroup')

    tree_nodes = tree.nodes
    tree_links = tree.links

    width = 200
    length = -1

    group_inputs = tree_nodes.new('NodeGroupInput')
    group_outputs = tree_nodes.new('NodeGroupOutput')

    for idx in ['i', 'j', 'l', 'm']:
        make_new_socket(tree, name=idx, io='INPUT', type='NodeSocketFloat')

    make_new_socket(tree, name='vi', io='INPUT', type='NodeSocketVector')
    make_new_socket(tree, name='vj', io='INPUT', type='NodeSocketVector')
    for i in range(5):
        make_new_socket(tree, name='s' + str(i), io='INPUT', type='NodeSocketFloat')

    make_new_socket(tree, name='intersection', io='OUTPUT', type='NodeSocketVector')

    group.name = name
    group.node_tree = tree
    group_inputs.location = (length * width, 0)
    length += 1

    pick_si = pick_by_index(tree_nodes, array_name='s', range=list(range(5)), name='si')
    pick_si.location = (length * width, -200)
    pick_si.hide = True
    tree_links.new(group_inputs.outputs['i'], pick_si.inputs['In'])
    for i in range(5):
        tree_links.new(group_inputs.outputs['s' + str(i)], pick_si.inputs['s' + str(i)])

    pick_sj = pick_by_index(tree_nodes, array_name='s', range=list(range(5)), name='sj')
    pick_sj.location = (length * width, -250)
    pick_sj.hide = True
    tree_links.new(group_inputs.outputs['j'], pick_sj.inputs['In'])
    for i in range(5):
        tree_links.new(group_inputs.outputs['s' + str(i)], pick_sj.inputs['s' + str(i)])

    length += 1

    det = create_group_from_vector_function(tree_nodes, functions=['vi_x,vj_y,*,vi_y,vj_x,*,-'],
                                            parameters=['vi', 'vj'],
                                            name='det', node_group_type='GeometryNodes')
    det.location = (length * width, 50)
    det.hide = True
    det.inputs[0].default_value = [0, 0, 1]  # take unit-z vector to pick the determinant in the cross-product

    for node, p in zip(['vi', 'vj'], ['vi', 'vj']):
        tree_links.new(group_inputs.outputs[node], det.inputs[p])
    length += 1

    scalars = ['det', 'l', 'm', 'si', 'sj']
    vectors = ['vi', 'vj']
    intersection_calc = create_group_from_vector_function(tree_nodes, functions=[
        "l,si,+,vj_y,*,m,sj,+,vi_y,*,-,det,/",
        "m,sj,+,vi_x,*,l,si,+,vj_x,*,-,det,/"
    ], parameters=vectors, scalar_parameters=scalars, name='IntersectionCalc', node_group_type='GeometryNodes')
    intersection_calc.location = (length * width, 0)
    length += 1

    for node, p in zip(['vi', 'vj'], ['vi', 'vj']):
        tree_links.new(group_inputs.outputs[node], intersection_calc.inputs[p])

    for i, s in enumerate(scalars):
        if i == 0:
            out = det.outputs['Out']
        elif i < 3:
            out = group_inputs.outputs[s]
        elif i == 3:
            out = pick_si.outputs[0]
        else:
            out = pick_sj.outputs[0]
        tree_links.new(out, intersection_calc.inputs[s])
    tree_links.new(intersection_calc.outputs[0], group_outputs.inputs[0])
    length += 1

    group_outputs.location = (length * width, 0)

    return group


def pick_vector_by_index(nodes, array_name='a', range=list(range(10)), name='a[i]'):
    a = array_name
    first = True
    part = ''
    for i in range:
        if first:
            part = ','.join([a + str(i), 'x', str(i), '=', 'scale'])
            first = False
        else:
            part = ','.join([part, a + str(i), 'x', str(i), '=', 'scale', 'add'])
    return create_group_from_vector_function(nodes, functions=[part], parameters=[array_name + str(i) for i in range],
                                             name=name,
                                             node_group_type='GeometryNodes')


def pick_by_index(nodes, array_name='a', range=list(range(10)), name='a[i]'):
    a = array_name
    first = True
    part = ''
    for i in range:
        if first:
            part = ','.join([a + str(i), 'x', str(i), '=', '*'])
            first = False
        else:
            part = ','.join([part, a + str(i), 'x', str(i), '=', '*', '+'])
    return create_group_from_scalar_function(nodes, functions=[part], parameters=[array_name + str(i) for i in range],
                                             name=name,
                                             node_group_type='GeometryNodes')


def create_index2ijlm_function(nodes, scalar_input_parameters=['index', 'base', 'range'],
                               name='Index2ijlm', node_tree_type='GeometryNodes'):
    """
       converts an index into a tuple of dimension dim
       with integer components in the range: -range ... range

       :param nodes:
       :param scalar_input_parameters: range=3, base =2*3+1, index = base**dim
       :param dim:
       :param name:
       :param node_tree_type:
       :return:
       """
    i_label = scalar_input_parameters[0]
    b_label = scalar_input_parameters[1]
    r_label = scalar_input_parameters[2]

    if node_tree_type == 'Shader':
        tree = bpy.data.node_groups.new(type='ShaderNodeTree', name=name)
        group = nodes.new(type='ShaderNodeGroup')
    else:
        tree = bpy.data.node_groups.new(type='GeometryNodeTree', name=name)
        group = nodes.new(type='GeometryNodeGroup')

    tree_nodes = tree.nodes
    tree_links = tree.links

    width = 200
    length = -1

    group_inputs = tree_nodes.new('NodeGroupInput')
    group_outputs = tree_nodes.new('NodeGroupOutput')

    for param in scalar_input_parameters:
        make_new_socket(tree, name=param, io='INPUT', type='NodeSocketFloat')

    make_new_socket(tree, name='i', io='OUTPUT', type='NodeSocketFloat')
    make_new_socket(tree, name='j', io='OUTPUT', type='NodeSocketFloat')
    make_new_socket(tree, name='l', io='OUTPUT', type='NodeSocketFloat')
    make_new_socket(tree, name='m', io='OUTPUT', type='NodeSocketFloat')

    group.name = name
    group.node_tree = tree
    group_inputs.location = (length * width, 0)
    length = 0
    i_socket = group_inputs.outputs[i_label]

    first = create_group_from_scalar_function(tree_nodes, ["x,b,b,*,/,floor"], parameters=['b'],
                                              name='idx//(2k+1)**2', node_group_type='GeometryNodes',
                                              output_names=["ij"])
    first.location = (length * width, 0)
    tree_links.new(group_inputs.outputs[i_label], first.inputs['In'])
    tree_links.new(group_inputs.outputs[b_label], first.inputs['b'])

    second = create_group_from_scalar_function(tree_nodes, ["x,b,b,*,%"], parameters=['b'], name='idx%(2k+1)**2',
                                               node_group_type='GeometryNodes', output_names=['lm'])
    second.location = (length * width, -200)
    tree_links.new(group_inputs.outputs[i_label], second.inputs['In'])
    tree_links.new(group_inputs.outputs[b_label], second.inputs['b'])

    length += 1

    ij_split = create_group_from_scalar_function(tree_nodes,
                                                 [
                                                     "x,3,>,x,6,>,+,x,8,>,+",
                                                     "x,4,<,1,x,4,%,+,*,x,3,>,x,7,<,*,2,x,4,%,+,*,+,x,6,>,x,9,<,*,3,x,7,%,+,*,+,x,8,>,4,*,+"
                                                 ], parameters=[], node_group_type='GeometryNodes',
                                                 output_names=['i', 'j'])

    ij_split.location = (length + width, 100)
    tree_links.new(first.outputs[0], ij_split.inputs['In'])
    tree_links.new(ij_split.outputs['i'], group_outputs.inputs[0])
    tree_links.new(ij_split.outputs['j'], group_outputs.inputs[1])

    lm_split = create_group_from_scalar_function(tree_nodes,
                                                 [
                                                     "x,b,/,floor,k,-",
                                                     "x,b,%,k,-"
                                                 ], parameters=['b', 'k'], node_group_type='GeometryNodes',
                                                 output_names=['l', 'm'])

    lm_split.location = (length + width, -200)
    tree_links.new(second.outputs[0], lm_split.inputs['In'])
    tree_links.new(lm_split.outputs['l'], group_outputs.inputs[2])
    tree_links.new(lm_split.outputs['m'], group_outputs.inputs[3])
    tree_links.new(group_inputs.outputs['base'], lm_split.inputs['b'])
    tree_links.new(group_inputs.outputs['range'], lm_split.inputs['k'])
    length += 1

    group_outputs.location = (length * width, 0)

    return group


def penrose_3D_analog(size=5, name="Penrose3DAnalog", radius=0.1, colors=None, plane_size=7):
    node_tree = setup_geometry_nodes(name)

    length = 0

    showPlane = InputBoolean(node_tree, location=(length, -3),
                             value=True,
                             name='showPlane', label='showPlane', hide=True)

    rotatePlane = InputVector(node_tree, location=(length, -3.25),
                              value=Vector([0, 0, 0]),
                              name='rotatePlane', label='rotatePlane', hide=True)

    unitX = InputVector(node_tree, location=(length, -3.5),
                        value=Vector([1, 0, 0]),
                        name='unitX', label='unitX', hide=True)
    unitY = InputVector(node_tree, location=(length, -3.75),
                        value=Vector([0, 1, 0]),
                        name='unitY', label='unitY', hide=True)
    unitZ = InputVector(node_tree, location=(length, -4),
                        value=Vector([0, 0, 1]),
                        name='unitZ', label='unitZ', hide=True)

    planeNormal = make_function(node_tree.nodes, functions={
        'normal': 'unitZ,rotatePlane,rot'},
                                inputs=['rotatePlane', 'unitZ'],
                                outputs=['normal'],
                                vectors=['rotatePlane', 'unitZ', 'normal'],
                                name="NormalOfPlane"
                                )
    planeNormal.location = ((length + 1) * 200, -3 * 200)
    planeNormal.hide = True
    node_tree.links.new(unitZ.std_out, planeNormal.inputs['unitZ'])
    node_tree.links.new(rotatePlane.std_out, planeNormal.inputs['rotatePlane'])

    planeOrigin = InputVector(node_tree, location=(length, -4.5),
                              value=Vector(),
                              name='planeOrigin', label='planeOrigin',
                              hide=True)

    mesh_line = MeshLine(node_tree, count=2 * size + 1, start_location=Vector([0, 0, -size]),
                         end_location=Vector([0, 0, size]),
                         location=(length, 0))
    length += 1

    grid = Grid(node_tree, location=(length, -0.5),
                size_x=2 * size, size_y=2 * size, vertices_x=2 * size + 1, vertices_y=2 * size + 1)
    length += 1

    instance_on_points = InstanceOnPoints(node_tree,
                                          location=(length, 0),
                                          instance=grid.outputs['Mesh'])
    length += 1

    realize_instances = RealizeInstances(node_tree, location=(length, 0.5))
    length += 1

    cubie_scale = InputValue(node_tree, location=(length - 1, -1.5), value=0.05, name="CubieScale")
    cubies = CubeMesh(node_tree, location=(length, -1.5),
                      size=cubie_scale.std_out)
    length += 1

    set_material = SetMaterial(node_tree, material='gray_5',
                               location=(length, -1.5))
    length += 1

    position = Position(node_tree,
                        location=add_locations(cubies.location, (0, 1.5)), hide=True)

    x_min = InputValue(node_tree,
                       location=add_locations(position.location, (-1, -0.25)),
                       hide=True, label='xMin', name='xMin', value=-50)
    x_max = InputValue(node_tree,
                       location=add_locations(position.location, (0, -0.25)),
                       hide=True, label='xMax', name='xMax', value=50)
    y_min = InputValue(node_tree,
                       location=add_locations(position.location, (-1, -0.5)),
                       hide=True, label='yMin', name='yMin', value=-50)
    y_max = InputValue(node_tree,
                       location=add_locations(position.location, (0, -0.5)),
                       hide=True, label='yMax', name='yMax', value=50)
    z_min = InputValue(node_tree,
                       location=add_locations(position.location, (-1, -0.75)),
                       hide=True, label='zMin', name='zMin', value=-50)
    z_max = InputValue(node_tree,
                       location=add_locations(position.location, (0, -0.75)),
                       hide=True, label='zMax', name='zMax', value=50)

    r_max = InputValue(node_tree,
                       location=add_locations(z_max.location, (0, -0.25)),
                       hide=True, label='rMax', name='rMax', value=0)

    nodes = [x_min, x_max, y_min, y_max, z_min, z_max, r_max, position]
    c_labels = ['x', 'y', 'z']
    minmax_values = flatten([[l + '_min', l + '_max'] for l in c_labels]) + ['r_max']
    all_labels = minmax_values + ['position']

    selector = make_function(node_tree.nodes, functions={
        'selector': ['position_x,x_min,>,position_x,x_max,<,*,' +
                     'position_y,y_min,>,*,position_y,y_max,<,*,' +
                     'position_z,z_min,>,*,position_z,z_max,<,*,' +
                     'position,length,r_max,<,*']
    }, name='Limits', inputs=all_labels,
                             outputs=['selector'],
                             vectors=['position'],
                             scalars=minmax_values + ['selector'])
    selector.hide = True
    selector.location = tuple([c * 200 for c in add_locations(set_material.location, (0, 1))])

    for node, label in zip(nodes, all_labels):
        node_tree.links.new(node.std_out, selector.inputs[label])

    create_geometry_line(node_tree, [cubies, set_material])

    instance_on_points2 = InstanceOnPoints(node_tree,
                                           location=(length, 0),
                                           selection=selector.outputs['selector'],
                                           instance=set_material.geometry_out)
    length += 1

    projection = InputValue(node_tree, name='Projection', label='Projection', value=0, location=(length, -1))
    projector = make_function(node_tree.nodes, functions={
        'V': 'v,planeNormal,v,planeOrigin,sub,planeNormal,dot,projection,*,scale,sub'
    }, name='Projector', inputs=['v', 'planeNormal', 'planeOrigin', 'projection'], outputs=['V'],
                              vectors=['v', 'planeNormal', 'planeOrigin', 'V'], scalars=['projection'])
    projector.location = (length * 200, 0)
    node_tree.links.new(projection.std_out, projector.inputs['projection'])
    node_tree.links.new(position.std_out, projector.inputs['v'])
    node_tree.links.new(planeNormal.outputs['normal'], projector.inputs['planeNormal'])
    node_tree.links.new(planeOrigin.std_out, projector.inputs['planeOrigin'])
    length += 1

    set_pos = SetPosition(node_tree,
                          location=(length, 0),
                          position=projector.outputs['V'])
    length += 2

    join = JoinGeometry(node_tree, location=(length, 0))
    length += 1

    shade_smooth = SetShadeSmooth(node_tree, location=(length, 0))
    length += 1

    join2 = JoinGeometry(node_tree, location=(length, 0))
    length += 1

    out = node_tree.nodes.get("Group Output")
    create_geometry_line(node_tree, [
        mesh_line,
        instance_on_points,
        realize_instances,
        instance_on_points2,
        set_pos,
        join,
        shade_smooth,
        join2
    ], out=out.inputs['Geometry'])

    # plane

    level = -3
    nlength = x_min.l
    point = Points(node_tree, location=(nlength, level + 1))
    plane = Grid(node_tree, location=(nlength, level), size_x=2 * plane_size, size_y=2 * plane_size,
                 vertices_x=2 * plane_size + 1, vertices_y=2 * plane_size + 1)
    nlength += 1
    wireframe = WireFrame(node_tree, location=(nlength, level), radius=0.02, resolution=4)
    nlength += 1

    set_material2 = SetMaterial(node_tree, location=(nlength, level),
                                material='plastic_custom1', roughness=0.1, name='planeMaterial', label='planeMaterial')
    nlength += 1
    create_geometry_line(node, [plane, wireframe, set_material2])

    instance3 = InstanceOnPoints(node_tree, location=(nlength, level + 1),
                                 instance=set_material2.geometry_out,
                                 selection=showPlane.std_out,
                                 rotation=rotatePlane.std_out)

    nlength += 1

    set_pos_plane = SetPosition(node_tree, location=(nlength, level + 1),
                                position=planeOrigin.std_out)
    nlength += 1

    # voronoi cell
    cube_scale = InputValue(node_tree, location=(-1, length / 4), value=1, name="CubeScale")
    cube = CubeMesh(node_tree, location=(0, length / 4))
    transform_geo = TransformGeometry(node_tree, location=(1, length / 4), translation=[0.5] * 3,
                                      scale=cube_scale.std_out)
    drawing = SetMaterial(node_tree, location=(2, length / 4), material='plastic_drawing',
                          roughness=0.25, name="CubeMaterial")
    create_geometry_line(node, [cube, transform_geo, drawing, join])

    # voronoi zone
    corner_vecs = [Vector(x) + Vector([0.5, 0.5, 0.5]) for x in tuples([0.5, -0.5], 3)]
    corners = [
        InputVector(node_tree, location=(-4, length / 3 - 0.25 * i), value=vec, name="Corner" + str(i), hide=True) for
        i, vec in enumerate(corner_vecs)]
    corner_args = ["v" + str(i) for i in range(len(corners))]

    min_max = make_function(node_tree, functions={
        "d_max": "v0,normal,dot,v1,normal,dot,max,v2,normal,dot,max,v3,normal,dot,max,v4,normal,dot,max,v5,normal,dot,max,v6,normal,dot,max,v7,normal,dot,max",
        "d_min": "v0,normal,dot,v1,normal,dot,min,v2,normal,dot,min,v3,normal,dot,min,v4,normal,dot,min,v5,normal,dot,min,v6,normal,dot,min,v7,normal,dot,min"
    }, inputs=corner_args + ["normal"], outputs=["d_min", "d_max"], vectors=corner_args + ["normal"],
                            scalars=["d_min", "d_max"], name="ZoneCalculator", hide=True)
    min_max.location = (-600, 200 * length / 3)
    for arg, node in zip(corner_args, corners):
        node_tree.links.new(node.std_out, min_max.inputs[arg])
    node_tree.links.new(planeNormal.outputs[0], min_max.inputs['normal'])

    # xy create faces
    scale = InputValue(node_tree, location=(length, -length / 4), name="ScaleElements", value=1)
    extrude = InputValue(node_tree, location=(length, -length / 4 - 0.25), name="ExtrudeElements", value=0)
    create_faces(node_tree, realize_instances, join, unitX, unitY, min_max, planeNormal, planeOrigin, projection, scale,
                 extrude, nlength, level, colors)
    create_faces(node_tree, realize_instances, join, unitX, unitZ, min_max, planeNormal, planeOrigin, projection, scale,
                 extrude, nlength, level - 1, colors)
    create_faces(node_tree, realize_instances, join, unitY, unitZ, min_max, planeNormal, planeOrigin, projection, scale,
                 extrude, nlength, level - 2, colors)

    create_geometry_line(node, [point, instance3, set_pos_plane, join])

    node_tree.nodes.get("Group Output").location = ((length + 2) * 200, 0)

    size = 7
    max_zone = Grid(node_tree, location=(-1, length / 4 - 1), size_x=2 * size, size_y=2 * size, vertices_x=2 * size + 1,
                    vertices_y=2 * size + 1, name="ZoneMax")
    min_zone = Grid(node_tree, location=(-1, length / 4 - 2), size_x=2 * size, size_y=2 * size, vertices_x=2 * size + 1,
                    vertices_y=2 * size + 1, name="ZoneMin")

    min_shift = make_function(node_tree, functions={
        "shift": "normal,d_min,scale"
    }, inputs=["normal", "d_min"], outputs=["shift"], scalars=["d_min"], vectors=["normal", "shift"],
                              name="MinShiftFunction", hide=True)
    node_tree.links.new(planeNormal.outputs[0], min_shift.inputs["normal"])
    node_tree.links.new(min_max.outputs["d_min"], min_shift.inputs["d_min"])
    min_shift.location = (-400, 200 * (length / 4 - 2))

    max_shift = make_function(node_tree, functions={
        "shift": "normal,d_max,scale"
    }, inputs=["normal", "d_max"], outputs=["shift"], scalars=["d_max"], vectors=["normal", "shift"],
                              name="MaxShiftFunction", hide=True)
    max_shift.location = (-400, 200 * (length / 4 - 1))
    node_tree.links.new(planeNormal.outputs[0], max_shift.inputs["normal"])
    node_tree.links.new(min_max.outputs["d_max"], max_shift.inputs["d_max"])

    shift_min = TransformGeometry(node_tree, location=(1, length / 4 - 2), translation=min_shift.outputs[0],
                                  rotation=rotatePlane.std_out,
                                  name="MinTransform")
    shift_max = TransformGeometry(node_tree, location=(1, length / 4 - 1), translation=max_shift.outputs[0],
                                  rotation=rotatePlane.std_out,
                                  name="MaxTransform")

    min_mat = SetMaterial(node_tree, material="plastic_drawing", location=(2, length / 4 - 2), name="MinMaterial")
    max_mat = SetMaterial(node_tree, material="plastic_drawing", location=(2, length / 4 - 1), name="MaxMaterial")
    min_wire = WireFrame(node_tree, location=(0, length / 4 - 2), radius=0.01, name="MinWireFrame")
    max_wire = WireFrame(node_tree, location=(0, length / 4 - 2), radius=0.01, name="MaxWireFrame")

    create_geometry_line(node_tree, [max_zone, max_wire, shift_max, max_mat, join])
    create_geometry_line(node_tree, [min_zone, min_wire, shift_min, min_mat, join])

    # selector
    selector2 = make_function(node_tree.nodes, functions={
        'select': 'pos,normal,dot,d_min,>,pos,normal,dot,d_max,>,not,*,limits,*'
    }, inputs=["normal", "pos", "d_min", "d_max", "limits"], outputs=["select"],
                              scalars=["limits", "d_max", "d_min", "select"], vectors=["normal", "pos"],
                              name="Selector", hide=True)
    node_tree.links.new(min_max.outputs["d_min"], selector2.inputs["d_min"])
    node_tree.links.new(min_max.outputs["d_max"], selector2.inputs["d_max"])
    node_tree.links.new(planeNormal.outputs[0], selector2.inputs["normal"])
    node_tree.links.new(position.std_out, selector2.inputs["pos"])
    node_tree.links.new(selector.outputs["selector"], selector2.inputs["limits"])
    selector.location = (-200, length / 5 * 200)

    ico_sphere = IcoSphere(node_tree, location=(0, length / 6), radius=0.075, subdivisions=2, name="SelSphere")
    ico_material = SetMaterial(node_tree, location=(3, length / 6), name="IcoSphereMaterial",
                               material="plastic_example")
    set_pos2 = SetPosition(node_tree, location=(4, length / 6), position=projector.outputs[0])
    instance4 = InstanceOnPoints(node_tree, instance=ico_sphere.geometry_out,
                                 location=(2, length / 5), selection=selector2.outputs['select'])

    create_geometry_line(node_tree, [realize_instances, instance4, ico_material, set_pos2, join2])

    return node_tree


def create_faces(node_tree, realize_instances, join, u, v, minmax_values, planeNormal, planeOrigin, projection, scale,
                 extrude, nlength, level, colors):
    uv_string = u.node.label[-1] + v.node.label[-1]
    range_name = uv_string.lower() + 'Range'
    if colors:
        dict = {
            'XY': colors[0],
            'XZ': colors[1],
            'YZ': colors[2],
        }
    else:
        dict = {
            'XY': 'joker',
            'XZ': 'important',
            'YZ': 'x23_color',
        }
    uv_range = InputValue(node_tree, name=range_name, label=range_name, value=10, location=(nlength, (level + 1)))
    face_uv = make_face_generator(node_tree.nodes,
                                  inputs=['Geometry', 'u', 'v', 'd_min', 'd_max', 'planeNormal', 'planeOrigin',
                                          'rLimit',
                                          'projection'],
                                  outputs=['Geometry'],
                                  input_types=['Geometry', 'Vector', 'Vector', 'Float', 'Float', 'Vector', 'Vector',
                                               'Float', 'Float'],
                                  output_types=['Geometry'], name='Faces' + uv_string)
    face_uv.hide = True
    face_uv.location = ((nlength + 1) * 200, (level + 1) * 200)
    node_tree.links.new(uv_range.std_out, face_uv.inputs['rLimit'])
    node_tree.links.new(realize_instances.outputs['Geometry'], face_uv.inputs['Geometry'])
    node_tree.links.new(u.std_out, face_uv.inputs['u'])
    node_tree.links.new(v.std_out, face_uv.inputs['v'])
    node_tree.links.new(minmax_values.outputs['d_min'], face_uv.inputs['d_min'])
    node_tree.links.new(minmax_values.outputs['d_max'], face_uv.inputs['d_max'])
    node_tree.links.new(planeNormal.outputs['normal'], face_uv.inputs['planeNormal'])
    node_tree.links.new(planeOrigin.std_out, face_uv.inputs['planeOrigin'])
    node_tree.links.new(projection.std_out, face_uv.inputs['projection'])

    uv_material = SetMaterial(node_tree, location=((nlength + 2), level + 2), material=dict[uv_string],
                              emission=0.015, rougness=0.4)
    node_tree.links.new(face_uv.outputs['Geometry'], uv_material.inputs['Geometry'])
    scale_element = ScaleElements(node_tree, location=((nlength + 3), level + 2), scale=scale.std_out)
    node_tree.links.new(uv_material.outputs['Geometry'], scale_element.inputs['Geometry'])
    extrude_element = ExtrudeMesh(node_tree, mode='FACES', location=((nlength + 4), level + 2), offset=extrude.std_out)
    node_tree.links.new(scale_element.geometry_out, extrude_element.geometry_in)
    node_tree.links.new(extrude_element.geometry_out, join.geometry_in)

    level -= 1


def make_face_generator(nodes, inputs=['Geometry'], outputs=['Geometry'],
                        input_types=['Geometry'], output_types=['Geometry'],
                        name='FaceGenerator', flip_orientation=False):
    tree = bpy.data.node_groups.new(type='GeometryNodeTree', name=name)
    group = nodes.new(type='GeometryNodeGroup')

    group.name = name
    group.node_tree = tree

    # create inputs and outputs
    tree_nodes = tree.nodes
    tree_links = tree.links

    group_inputs = tree_nodes.new('NodeGroupInput')
    group_outputs = tree_nodes.new('NodeGroupOutput')

    for ins, in_type in zip(inputs, input_types):
        socket_type = 'NodeSocket' + in_type
        make_new_socket(tree, name=ins, io='INPUT', type=socket_type)

    for outs, out_type in zip(outputs, output_types):
        socket_type = 'NodeSocket' + out_type
        make_new_socket(tree, name=outs, io='OUTPUT', type=socket_type)

    group_inputs.location = (-400, 0)
    group_outputs.location = (6 * 200, 0)

    points = Points(tree, location=(2, -1), count=1)

    position = Position(tree, location=(-1, -1))
    position.hide = True

    projection = make_function(tree_nodes, functions={
        'P': 'position,planeNormal,position,planeOrigin,sub,planeNormal,dot,projection,*,scale,sub',
        'U': 'u,planeNormal,u,planeNormal,dot,projection,*,scale,sub',
        'V': 'v,planeNormal,v,planeNormal,dot,projection,*,scale,sub'
    },
                               inputs=['position', 'u', 'v', 'planeNormal', 'planeOrigin', 'projection'],
                               outputs=['P', 'U', 'V'],
                               vectors=['position', 'u', 'v', 'planeNormal', 'planeOrigin', 'P', 'U', 'V'],
                               scalars=['projection'],
                               name='ProjectorOf' + name
                               )
    projection.location = (200, 200)

    tree_links.new(group_inputs.outputs['u'], projection.inputs['u'])
    tree_links.new(group_inputs.outputs['v'], projection.inputs['v'])

    tree_links.new(group_inputs.outputs['planeNormal'], projection.inputs['planeNormal'])
    tree_links.new(group_inputs.outputs['planeOrigin'], projection.inputs['planeOrigin'])
    tree_links.new(position.std_out, projection.inputs['position'])
    tree_links.new(group_inputs.outputs['projection'], projection.inputs['projection'])

    selector = make_function(tree_nodes, functions={
        'selector': 'position,length,rLimit,<,' +
                    'position,planeOrigin,sub,planeNormal,dot,d_min,>,position,planeOrigin,sub,planeNormal,dot,d_max,>,not,*,*,' +
                    'position,u,add,planeOrigin,sub,planeNormal,dot,d_min,>,position,u,add,planeOrigin,sub,planeNormal,dot,d_max,>,not,*,*,' +
                    'position,v,add,planeOrigin,sub,planeNormal,dot,d_min,>,position,v,add,planeOrigin,sub,planeNormal,dot,d_max,>,not,*,*,' +
                    'position,u,add,v,add,planeOrigin,sub,planeNormal,dot,d_min,>,position,u,add,v,add,planeOrigin,sub,planeNormal,dot,d_max,>,not,*,*'
    },
                             inputs=['position', 'u', 'v', 'd_min', 'd_max', 'planeNormal', 'planeOrigin', 'rLimit'],
                             outputs=['selector'],
                             vectors=['position', 'u', 'v', 'planeNormal', 'planeOrigin'],
                             scalars=['rLimit', 'd_min', 'd_max', 'selector'],
                             name='SelectorOf' + name
                             )
    selector.location = (200, -100)

    tree_links.new(position.std_out, selector.inputs['position'])
    for label in ['u', 'v', 'd_min', 'd_max', 'planeNormal', 'planeOrigin', 'rLimit']:
        tree_links.new(group_inputs.outputs[label], selector.inputs[label])

    instance = InstanceOnPoints(tree, location=(2, 0),
                                instance=points.geometry_out,
                                selection=selector.outputs['selector'])

    set_pos = SetPosition(tree, location=(3, 0),
                          position=projection.outputs['P'])
    p2v = PointsToVertices(tree, location=(4, 0))

    extrudePoints = ExtrudeMesh(tree, mode='VERTICES', location=(5, 0),
                                offset=projection.outputs['U'])
    extrudeEdges = ExtrudeMesh(tree, mode='EDGES', location=(6, 0),
                               offset=projection.outputs['V'])

    create_geometry_line(tree, [instance, set_pos, p2v, extrudePoints, extrudeEdges],
                         ins=group_inputs.outputs['Geometry'],
                         out=group_outputs.inputs['Geometry'])
    return group
