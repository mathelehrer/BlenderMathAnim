import numpy as np

from appearance.textures import phase2hue_material
from geometry_nodes.nodes import layout, Points, InputValue, CurveCircle, InstanceOnPoints, JoinGeometry, \
    create_geometry_line, RealizeInstances, Position, make_function, ObjectInfo, SetPosition, Index, SetMaterial, \
    RandomValue, RepeatZone, StoredNamedAttribute, NamedAttribute, VectorMath, CurveToMesh, PointsToCurve, Grid, \
    TransformGeometry, InputVector, DeleteGeometry, IcoSphere, MeshLine, MeshToCurve, InstanceOnEdges, CubeMesh, \
    EdgeVertices, BooleanMath, SetShadeSmooth, RayCast, WireFrame, ConvexHull, InsideConvexHull, ExtrudeMesh, \
    ScaleElements, UVSphere
from interface import ibpy
from interface.ibpy import make_new_socket, Vector, get_node_tree
from mathematics.parsing.parser import ExpressionConverter
from mathematics.spherical_harmonics import SphericalHarmonics
from objects.derived_objects.p_arrow import PArrow

pi = np.pi
r2 = np.sqrt(2)
r3 = np.sqrt(3)


class GeometryNodesModifier:
    """
    base class that organizes the boilerplate code for the creation of a geometry nodes modifier
    """

    def __init__(self, name='DefaultGeometryNodeGroup', automatic_layout=True):
        tree = get_node_tree(name=name, type='GeometryNodeTree')

        # if  materials are created inside the geometry node, they are stored inside the following array
        # and will be added to the material slots of the blender object
        self.materials = []
        # create output nodes
        self.group_outputs = tree.nodes.new('NodeGroupOutput')
        make_new_socket(tree, name='Geometry', io='OUTPUT', type='NodeSocketGeometry')
        self.create_node(tree)
        # automatically layout nodes
        if automatic_layout:
            layout(tree)
        self.tree = tree

    def create_node(self, tree):
        """
        this method is to be overridden to customize your geometry node modifier
        :param nodes:
        :param links:
        :return:
        """
        pass

    def get_node_tree(self):
        return self.tree


class VectorLogo(GeometryNodesModifier):
    def __init__(self, name='VectorLogo', n=10, colors=['important', 'example', 'drawing']):
        self.n = n
        self.colors = colors
        super().__init__(name)

    def create_node(self, tree):
        # arrow object with origin at the tip
        arrow = PArrow(name="ArrowObject")
        ibpy.set_pivot(arrow, Vector([0, 0, 1]))

        # parameters for animations
        res = InputValue(tree, name='res', value=10)
        n = InputValue(tree, name='N', value=self.n)
        idx = Index(tree)
        growth = InputValue(tree, name='growth', value=0)

        object_info = ObjectInfo(tree, transform_space='ORIGINAL', object=arrow.ref_obj)
        pos = Position(tree)

        # function for the red circles
        red_circles = make_function(tree.nodes, functions={
            "center": ["1,2,index,N,2,/,floor,-,2,**,+,/,2,*,index,N,2,/,floor,-,*",
                       "1,2,index,N,2,/,floor,-,2,**,+,/,3,*", "0"],
            "radius": "1,2,index,N,2,/,floor,-,2,**,+,/"
        }, inputs=['N', 'index'], outputs=['center', 'radius'],
                                    scalars=['N', 'index', 'radius'], vectors=['center'], name="RedCircles")
        tree.links.new(n.std_out, red_circles.inputs['N'])
        tree.links.new(idx.std_out, red_circles.inputs['index'])

        cloud = Points(tree, name='PointCloud', position=red_circles.outputs["center"], count=n.std_out)

        circle = CurveCircle(tree, resolution=res.std_out, radius=1)
        mul_red = make_function(tree.nodes, functions={
            "s": "radius,growth,*"
        }, inputs=['radius', 'growth'], outputs=['s'], scalars=['radius', 'growth', 's'])
        tree.links.new(growth.std_out, mul_red.inputs['growth'])
        tree.links.new(red_circles.outputs['radius'], mul_red.inputs['radius'])
        instance = InstanceOnPoints(tree, instance=circle.geometry_out, scale=mul_red.outputs['s'])

        rotation = make_function(tree.nodes, functions={
            "Rotation": "pos,e_z,cross,pi,-2,/,axis_rot,rot2euler"
        }, name="Rotation", inputs=["pos"], outputs=["Rotation"], vectors=["pos", "Rotation"])
        instance2 = InstanceOnPoints(tree, instance=object_info.geometry_out)
        tree.links.new(rotation.outputs['Rotation'], instance2.inputs['Rotation'])
        tree.links.new(pos.std_out, rotation.inputs['pos'])
        material = SetMaterial(tree, material=self.colors[0])
        join = JoinGeometry(tree)
        create_geometry_line(tree, [cloud, instance, instance2, material, join],
                             out=self.group_outputs.inputs['Geometry'])

        # yellow circles
        n_string = "index,N,2,/,floor,-"
        r_string = "1,6,4," + n_string + ",*," + n_string + ",1,-,*,+,/"
        yellow_circles = make_function(tree.nodes, functions={
            "center": ["8," + n_string + ",*,4,-," + r_string + ",*", "9," + r_string + ",*", "0"],
            "radius": r_string
        }, inputs=['N', 'index'], outputs=['center', 'radius'],
                                       scalars=['N', 'index', 'radius'], vectors=['center'], name="YellowCircles")
        tree.links.new(n.std_out, yellow_circles.inputs['N'])
        tree.links.new(idx.std_out, yellow_circles.inputs['index'])

        cloud = Points(tree, name='PointCloud', position=yellow_circles.outputs["center"], count=n.std_out)
        circle = CurveCircle(tree, resolution=res.std_out, radius=1)
        mul_yellow = make_function(tree.nodes, functions={
            "s": "radius,growth,*"
        }, inputs=['radius', 'growth'], outputs=['s'], scalars=['radius', 'growth', 's'])
        tree.links.new(growth.std_out, mul_yellow.inputs['growth'])
        tree.links.new(yellow_circles.outputs['radius'], mul_yellow.inputs['radius'])
        instance = InstanceOnPoints(tree, instance=circle.geometry_out, scale=mul_yellow.outputs['s'])

        # rotation = make_function(tree.nodes, functions={
        #     "Rotation": "pos,e_z,cross,pi,-2,/,axis_rot,rot2euler"
        # }, name="Rotation", inputs=["pos"], outputs=["Rotation"], vectors=["pos", "Rotation"])
        instance2 = InstanceOnPoints(tree, instance=object_info.geometry_out)
        tree.links.new(rotation.outputs['Rotation'], instance2.inputs['Rotation'])
        tree.links.new(pos.std_out, rotation.inputs['pos'])
        material = SetMaterial(tree, material=self.colors[1])
        create_geometry_line(tree, [cloud, instance, instance2, material, join],
                             out=self.group_outputs.inputs['Geometry'])

        # blue circles
        n_string = "index,N,2,/,floor,-"
        r_string = "1,15,4," + n_string + ",*," + n_string + ",1,-,*,+,/"
        blue_circles = make_function(tree.nodes, functions={
            "center": ["8," + n_string + ",*,4,-," + r_string + ",*", "15," + r_string + ",*", "0"],
            "radius": r_string
        }, inputs=['N', 'index'], outputs=['center', 'radius'],
                                     scalars=['N', 'index', 'radius'], vectors=['center'], name="BlueCircles")
        tree.links.new(n.std_out, blue_circles.inputs['N'])
        tree.links.new(idx.std_out, blue_circles.inputs['index'])

        cloud = Points(tree, name='PointCloud', position=blue_circles.outputs["center"], count=n.std_out)
        circle = CurveCircle(tree, resolution=res.std_out, radius=1)
        mul_blue = make_function(tree.nodes, functions={
            "s": "radius,growth,*"
        }, inputs=['radius', 'growth'], outputs=['s'], scalars=['radius', 'growth', 's'])
        tree.links.new(growth.std_out, mul_blue.inputs['growth'])
        tree.links.new(blue_circles.outputs['radius'], mul_blue.inputs['radius'])
        instance = InstanceOnPoints(tree, instance=circle.geometry_out, scale=mul_blue.outputs['s'])

        # rotation = make_function(tree.nodes, functions={
        #     "Rotation": "pos,e_z,cross,pi,-2,/,axis_rot,rot2euler"
        # }, name="Rotation", inputs=["pos"], outputs=["Rotation"], vectors=["pos", "Rotation"])
        instance2 = InstanceOnPoints(tree, instance=object_info.geometry_out)
        tree.links.new(rotation.outputs['Rotation'], instance2.inputs['Rotation'])
        tree.links.new(pos.std_out, rotation.inputs['pos'])
        material = SetMaterial(tree, material=self.colors[2])
        create_geometry_line(tree, [cloud, instance, instance2, material, join],
                             out=self.group_outputs.inputs['Geometry'])

        self.arrow = arrow

    def get_arrow_object(self):
        return self.arrow


class LorentzAttractorNode(GeometryNodesModifier):
    def __init__(self, name='LorentzAttractor', iterations=15000, a=0.4):
        self.iterations = iterations
        self.a = a
        super().__init__(name)

    def create_node(self, tree):
        random_value = RandomValue(tree, data_type='FLOAT_VECTOR', min=-0.5 * Vector([1, 1, 1]),
                                   max=0.5 * Vector([1, 1, 1]))
        position = Position(tree)
        a_value = InputValue(tree, value=1.4)
        points = Points(tree, position=random_value.std_out)
        repeat = RepeatZone(tree, iterations=self.iterations, hide=False)
        repeat.add_socket(socket_type="GEOMETRY", name="Joined")
        repeat.geometry_out = repeat.repeat_output.outputs[
            "Joined"]  # make the second socket the default geometry output
        repeat.join_in_geometries(out_socket_name='Joined')

        namedAttribute = NamedAttribute(tree, data_type='FLOAT_VECTOR', name='pos')

        storedNamedAttr = StoredNamedAttribute(tree, data_type='FLOAT_VECTOR', name='pos')
        transformation = make_function(tree.nodes, functions={
            "position": [
                "old_pos_x,a,pos_x,*,-,4,pos_y,*,-,4,pos_z,*,-,pos_y,2,**,-",
                "old_pos_y,a,pos_y,*,-,4,pos_z,*,-,4,pos_x,*,-,pos_z,2,**,-",
                "old_pos_z,a,pos_z,*,-,4,pos_x,*,-,4,pos_y,*,-,pos_x,2,**,-",
            ]
        }, inputs=["pos", "old_pos", 'a'], outputs=["position"], vectors=["pos", "old_pos", "position"], scalars=['a'])
        tree.links.new(namedAttribute.std_out, transformation.inputs['old_pos'])
        tree.links.new(position.std_out, transformation.inputs['pos'])
        tree.links.new(transformation.outputs['position'], storedNamedAttr.inputs["Value"])
        tree.links.new(a_value.std_out, transformation.inputs['a'])
        namedAttribute2 = NamedAttribute(tree, data_type='FLOAT_VECTOR', name='pos')
        scale = VectorMath(tree, operation='SCALE', float_input=0.005, inputs0=namedAttribute2.std_out)
        setPosition = SetPosition(tree, position=scale.std_out)

        repeat.create_geometry_line([storedNamedAttr, setPosition])

        points2curve = PointsToCurve(tree)
        circle = CurveCircle(tree, resolution=8, radius=0.3)
        curve2mesh = CurveToMesh(tree, profile_curve=circle.geometry_out)

        create_geometry_line(tree, [points, repeat, points2curve, curve2mesh],
                             out=self.group_outputs.inputs['Geometry'])
        self.repeat = repeat

    def get_iteration_socket(self):
        return self.repeat.repeat_input.inputs[0]


class Penrose2DIntro(GeometryNodesModifier):
    def __init__(self, name='Penrose2D'):
        super().__init__(name)

    def create_node(self, tree):
        # parameters
        angle = InputValue(tree, value=pi / 2, name="Angle")
        lattice_thickness = InputValue(tree, value=0.1, name="Thickness")
        visible_radius = InputValue(tree, value=0, name="GrowGrid")
        visible_length = InputValue(tree, value=0, name="GrowLine")

        position = Position(tree)
        projector = InputValue(tree, value=0, name='Projector')

        # create voronoi zone
        normal = make_function(tree, functions={
            "normal": ["angle,cos", "angle,sin", "0"]
        }, inputs=['angle'], outputs=['normal'], scalars=['angle'], vectors=['normal'],
                               name='Normal', hide=True)
        tree.links.new(angle.std_out, normal.inputs['angle'])

        # projection
        projection = make_function(tree, functions={
            "location": "position,normal,normal,position,dot,projector,*,scale,sub"
        }, inputs=['position', 'projector', 'normal'], outputs=['location'], scalars=['projector'],
                                   vectors=['normal', 'position', 'location'], name="Projection", hide=True)
        tree.links.new(normal.outputs['normal'], projection.inputs['normal'])
        tree.links.new(position.std_out, projection.inputs['position'])
        tree.links.new(projector.std_out, projection.inputs['projector'])

        # grid

        mesh = Grid(tree, size_x=30, size_y=20, vertices_y=21, vertices_x=31)
        join_geometry = JoinGeometry(tree)

        cube = CubeMesh(tree, size=[0.1] * 3)
        instance_on_points = InstanceOnPoints(tree, instance=cube.geometry_out)
        selector = make_function(tree, functions={
            'select': "pos,length,r,>"
        }, name="SelectorGrid", inputs=['pos', 'r'], outputs=['select'], scalars=['r', 'select'], vectors=['pos'])
        tree.links.new(visible_radius.std_out, selector.inputs['r'])
        tree.links.new(position.std_out, selector.inputs['pos'])

        set_projection = SetPosition(tree, position=projection.outputs['location'])

        delete_geom = DeleteGeometry(tree, selection=selector.outputs['select'])
        create_geometry_line(tree, [mesh, delete_geom, set_projection, instance_on_points, join_geometry],
                             out=self.group_outputs.inputs[0])

        instance_on_edges = InstanceOnEdges(tree, radius=lattice_thickness.std_out, name="Lattice")
        mat = SetMaterial(tree, material='joker')
        create_geometry_line(tree, [set_projection, instance_on_edges, mat, join_geometry])

        # mesh line branch
        rotation = make_function(tree.nodes, functions={
            "rotation": ["pi,2,/", "0", "angle"]
        }, inputs=['angle'], outputs=['rotation'], scalars=['angle'], vectors=['rotation'], name="Rotation", hide=True)
        tree.links.new(angle.std_out, rotation.inputs['angle'])

        selector2 = make_function(tree, functions={
            'select': "pos,length,r,>"
        }, name="SelectorLine", inputs=['pos', 'r'], outputs=['select'], scalars=['r', 'select'], vectors=['pos'])
        tree.links.new(visible_length.std_out, selector2.inputs['r'])
        tree.links.new(position.std_out, selector2.inputs['pos'])

        mesh_line = MeshLine(tree, name="ProjectionLine", count=1000,
                             start_location=Vector([0, 0, -15]), end_location=Vector([0, 0, 15]))
        delete_line = DeleteGeometry(tree, selection=selector2.outputs['select'])
        instance_on_edge = InstanceOnEdges(tree, radius=0.025, resolution=8)
        transform_geometry2 = TransformGeometry(tree, rotation=rotation.outputs['rotation'])
        mat_line = SetMaterial(tree, material='plastic_custom1')

        create_geometry_line(tree,
                             [mesh_line, delete_line, instance_on_edge, transform_geometry2, mat_line, join_geometry])

    def get_iteration_socket(self):
        return self.repeat.repeat_input.inputs[0]


class Penrose2DVoronoi(GeometryNodesModifier):
    def __init__(self, name='Penrose2D'):
        super().__init__(name)

    def create_node(self, tree):
        # parameters
        angle = InputValue(tree, value=pi / 2, name="Angle")
        lattice_thickness = InputValue(tree, value=0, name="Thickness")
        visible_radius = InputValue(tree, value=0, name="GrowGrid")
        visible_length = InputValue(tree, value=0, name="GrowLine")

        dl = InputVector(tree, value=Vector([0, 0, 0]), name="DL")
        dr = InputVector(tree, value=Vector([0, 1, 0]), name="DR")
        ul = InputVector(tree, value=Vector([1, 0, 0]), name="UL")
        ur = InputVector(tree, value=Vector([1, 1, 0]), name='UR')

        position = Position(tree)
        projector = InputValue(tree, value=0, name='Projector')

        shift = InputVector(tree, value=Vector(), name='VoronoiShift')
        offset = InputVector(tree, value=Vector(), name='GridShift')

        # normal
        normal = make_function(tree, functions={
            "normal": ["angle,cos", "angle,sin", "0"]
        }, inputs=['angle'], outputs=['normal'], scalars=['angle'], vectors=['normal'],
                               name='Normal', hide=True)
        tree.links.new(angle.std_out, normal.inputs['angle'])

        # projection
        projection = make_function(tree, functions={
            "location": "position,normal,normal,position,dot,projector,*,scale,sub"
        }, inputs=['position', 'projector', 'normal'], outputs=['location'], scalars=['projector'],
                                   vectors=['normal', 'position', 'location'], name="Projection", hide=True)
        tree.links.new(normal.outputs['normal'], projection.inputs['normal'])
        tree.links.new(position.std_out, projection.inputs['position'])
        tree.links.new(projector.std_out, projection.inputs['projector'])

        # grid

        mesh = Grid(tree, size_x=40, size_y=20, vertices_y=21, vertices_x=41)
        offset_transform = TransformGeometry(tree, translation=offset.std_out, name="GridShift")

        join_geometry = JoinGeometry(tree)

        cube = CubeMesh(tree, size=[0.1] * 3)
        instance_on_points = InstanceOnPoints(tree, instance=cube.geometry_out)

        set_projection = SetPosition(tree, position=projection.outputs['location'])

        instance_on_edges = InstanceOnEdges(tree, radius=lattice_thickness.std_out, name="Lattice")
        mat = SetMaterial(tree, material='joker', name="LatticeMaterial")

        create_geometry_line(tree, [set_projection, instance_on_edges, mat, join_geometry])

        # mesh line branch
        rotation = make_function(tree.nodes, functions={
            "rotation": ["pi,2,/", "0", "angle"]
        }, inputs=['angle'], outputs=['rotation'], scalars=['angle'], vectors=['rotation'],
                                 name="Rotation", hide=True)
        tree.links.new(angle.std_out, rotation.inputs['angle'])

        selector2 = make_function(tree, functions={
            'select': "pos,length,r,>"
        }, name="SelectorLine", hide=True, inputs=['pos', 'r'], outputs=['select'],
                                  scalars=['r', 'select'], vectors=['pos'])
        tree.links.new(visible_length.std_out, selector2.inputs['r'])
        tree.links.new(position.std_out, selector2.inputs['pos'])

        mesh_line = MeshLine(tree, name="ProjectionLine", count=30,
                             start_location=Vector([0, 0, -15]), end_location=Vector([0, 0, 15]))
        instance_on_edge = InstanceOnEdges(tree, radius=0.025, resolution=8)
        transform_geometry2 = TransformGeometry(tree, rotation=rotation.outputs['rotation'])
        mat_line = SetMaterial(tree, material='plastic_custom1', name="ProjectionMaterial")

        create_geometry_line(tree, [mesh_line, instance_on_edge, transform_geometry2, mat_line, join_geometry])

        # create voronoi zone

        voronoi = make_function(tree, functions={
            "min": [
                "dl,shift,add,normal,dot,dr,shift,add,normal,dot,min,ul,shift,add,normal,dot,min,ur,shift,add,normal,dot,min"],
            "max": [
                "dl,shift,add,normal,dot,dr,shift,add,normal,dot,max,ul,shift,add,normal,dot,max,ur,shift,add,normal,dot,max"],
        }, inputs=['dl', 'dr', 'ul', 'ur', 'normal', 'shift'], outputs=['min', 'max'],
                                scalars=['min', 'max'], vectors=['shift', 'dl', 'dr', 'ul', 'ur', 'normal'],
                                name='Voronoi',
                                hide=True)
        tree.links.new(normal.outputs['normal'], voronoi.inputs['normal'])
        tree.links.new(dl.std_out, voronoi.inputs['dl'])
        tree.links.new(dr.std_out, voronoi.inputs['dr'])
        tree.links.new(ur.std_out, voronoi.inputs['ur'])
        tree.links.new(ul.std_out, voronoi.inputs['ul'])
        tree.links.new(shift.std_out, voronoi.inputs['shift'])

        # implement min< n*pos <= max
        close_to = make_function(tree, functions={
            "is": ["position,normal,dot,minimum,>,position,normal,dot,maximum,>,not,*"],
            "is_not": ["position,normal,dot,minimum,>,not,position,normal,dot,maximum,>,+"]
        }, name="CloseTo", inputs=['position', 'minimum', 'maximum', 'normal'], outputs=['is', 'is_not'],
                                 scalars=['minimum', 'maximum', 'is', 'is_not'],
                                 vectors=['position', 'normal'], hide=True)
        tree.links.new(position.std_out, close_to.inputs['position'])
        tree.links.new(normal.outputs['normal'], close_to.inputs['normal'])
        tree.links.new(voronoi.outputs['min'], close_to.inputs['minimum'])
        tree.links.new(voronoi.outputs['max'], close_to.inputs['maximum'])

        # finish grid line

        set_pos_for_inside_points = SetPosition(tree, selection=close_to.outputs["is"],
                                                position=projection.outputs["location"])
        grid_material = SetMaterial(tree, material='gray_5', name="GridMaterial")
        create_geometry_line(tree,
                             [mesh, offset_transform, set_pos_for_inside_points, instance_on_points, grid_material,
                              join_geometry],
                             out=self.group_outputs.inputs[0])

        # voronoi cell
        voronoi_cell = Grid(tree, name='VoronoiCell', size_x=1, size_y=1, vertices_x=2, vertices_y=2)
        voronoi_shift = make_function(tree, functions={
            "translation": "e_x,0.5,scale,e_y,0.5,scale,add,shift,add"
        }, inputs=['shift'], outputs=['translation'], vectors=['shift', 'translation'], hide=True, name='VoronoiShift')
        tree.links.new(shift.std_out, voronoi_shift.inputs['shift'])
        transform_geometry3 = TransformGeometry(tree, translation=voronoi_shift.outputs['translation'])
        mat_cell = SetMaterial(tree, material='plastic_drawing', name="VoronoiCellMaterial")
        create_geometry_line(tree, [voronoi_cell, transform_geometry3, mat_cell, join_geometry])

        # voronoi zone

        mesh_line_max = MeshLine(tree, name='SelectionZoneMax', count=1000,
                                 start_location=Vector([0, 0, -15]), end_location=Vector([0, 0, 15]))
        mesh_line_min = MeshLine(tree, name='SelectionZoneMin', count=1000,
                                 start_location=Vector([0, 0, -15]), end_location=Vector([0, 0, 15]))

        instance_on_edge_max = InstanceOnEdges(tree, resolution=4, radius=0.01)
        instance_on_edge_min = InstanceOnEdges(tree, resolution=4, radius=0.01)

        translation_max = make_function(tree, functions={
            "translation": "normal,maximum,scale"
        }, inputs=['normal', 'maximum'], outputs=['translation'], scalars=['maximum'],
                                        vectors=['normal', 'translation'], name='TransMax', hide=True
                                        )
        tree.links.new(normal.outputs['normal'], translation_max.inputs['normal'])
        tree.links.new(voronoi.outputs['max'], translation_max.inputs['maximum'])

        translation_min = make_function(tree, functions={
            "translation": "normal,minimum,scale"
        }, inputs=['normal', 'minimum'], outputs=['translation'], scalars=['minimum'],
                                        vectors=['normal', 'translation'], name='TransMin', hide=True
                                        )
        tree.links.new(normal.outputs['normal'], translation_min.inputs['normal'])
        tree.links.new(voronoi.outputs['min'], translation_min.inputs['minimum'])

        transform_geometry_max = TransformGeometry(tree, rotation=rotation.outputs['rotation'],
                                                   translation=translation_max.outputs['translation'])
        transform_geometry_min = TransformGeometry(tree, rotation=rotation.outputs['rotation'],
                                                   translation=translation_min.outputs['translation'])
        mat_max = SetMaterial(tree, material='plastic_drawing', name="MaxMaterial")
        mat_min = SetMaterial(tree, material='plastic_drawing', name="MinMaterial")

        delete_line_max = DeleteGeometry(tree, selection=selector2.outputs['select'])
        delete_line_min = DeleteGeometry(tree, selection=selector2.outputs['select'])
        create_geometry_line(tree,
                             [mesh_line_max, delete_line_max, instance_on_edge_max, transform_geometry_max, mat_max,
                              join_geometry])
        create_geometry_line(tree,
                             [mesh_line_min, delete_line_min, instance_on_edge_min, transform_geometry_min, mat_min,
                              join_geometry])

        # highlight inside zone:
        # inside line
        selector = make_function(tree, functions={
            'select': "pos,length,r,>"
        }, name="SelectorGrid", inputs=['pos', 'r'], outputs=['select'],
                                 scalars=['r', 'select'], vectors=['pos'], hide=True)
        tree.links.new(visible_radius.std_out, selector.inputs['r'])
        tree.links.new(position.std_out, selector.inputs['pos'])

        delete_geometry = DeleteGeometry(tree, selection=close_to.outputs["is_not"])
        delete_geometry2 = DeleteGeometry(tree, selection=selector.outputs["select"])

        icosphere = IcoSphere(tree, radius=0.1, subdivisions=2, name="SelectedPoints")
        instance_on_points = InstanceOnPoints(tree, instance=icosphere.geometry_out, selection=close_to.outputs["is"])
        set_pos_icos = SetPosition(tree, position=projection.outputs['location'])
        icos_mat = SetMaterial(tree, material='plastic_example', name="IcosphereMaterial")
        create_geometry_line(tree, [offset_transform, delete_geometry,
                                    instance_on_points, set_pos_icos, icos_mat, join_geometry],
                             out=self.group_outputs.inputs[0])

        # split horizontal and vertical lines
        edge_vertices = EdgeVertices(tree)
        horizontal_tester = make_function(tree, functions={
            "is": "pos1,pos2,sub,e_x,dot,abs,0,>",
        }, inputs=["pos1", "pos2"], outputs=["is", "is_not"], vectors=["pos1", "pos2"], scalars=["is"],
                                          name="HorizontalTester", hide=True)
        tree.links.new(edge_vertices.outputs["Position 1"], horizontal_tester.inputs["pos1"])
        tree.links.new(edge_vertices.outputs["Position 2"], horizontal_tester.inputs["pos2"])

        stored_attribute = StoredNamedAttribute(tree, data_type='BOOLEAN', domain='EDGE', name='horizontalAttribute',
                                                value=horizontal_tester.outputs[0])

        named_attribute = NamedAttribute(tree, data_type='BOOLEAN', name='horizontalAttribute')
        not_node = BooleanMath(tree, operation='NOT', inputs0=named_attribute.std_out, name="Not")

        # horizontal
        instance_on_edges_horizontal = InstanceOnEdges(tree, selection=named_attribute.std_out, resolution=8,
                                                       radius=0.05, name="Horizontal")
        set_pos_horizontal = SetPosition(tree, position=projection.outputs['location'])
        horizontal_mat = SetMaterial(tree, selection=None, material='plastic_joker', name="HorizontalMaterial")

        # vertical
        instance_on_edges_vertical = InstanceOnEdges(tree, selection=not_node.std_out, resolution=8, radius=0.05,
                                                     name="Vertical")
        set_pos_vertical = SetPosition(tree, position=projection.outputs['location'])
        vertical_mat = SetMaterial(tree, selection=None, material='plastic_important', name="VerticalMaterial")

        create_geometry_line(tree, [offset_transform, delete_geometry, stored_attribute, delete_geometry2,
                                    set_pos_horizontal,
                                    instance_on_edges_horizontal, horizontal_mat, join_geometry],
                             out=self.group_outputs.inputs[0])
        create_geometry_line(tree, [stored_attribute, delete_geometry2, set_pos_vertical, instance_on_edges_vertical,
                                    vertical_mat,
                                    join_geometry])

    def get_iteration_socket(self):
        return self.repeat.repeat_input.inputs[0]


class Penrose2D(GeometryNodesModifier):
    def __init__(self, name='Penrose2D'):
        super().__init__(name)

    def create_node(self, tree):
        # parameters
        angle = InputValue(tree, value=0, name="angle")
        offset = InputVector(tree, value=Vector([0.2, 0.2, 0]), name='Offset')
        dl = InputVector(tree, value=Vector([0, 0, 0]), name="DL")
        dr = InputVector(tree, value=Vector([0, 1, 0]), name="DR")
        ul = InputVector(tree, value=Vector([1, 0, 0]), name="UL")
        ur = InputVector(tree, value=Vector([1, 1, 0]), name='UR')
        position = Position(tree)
        projector = InputValue(tree, value=0, name='Projector')

        # create voronoi zone
        normal = make_function(tree, functions={
            "normal": ["angle,cos", "angle,sin", "0"]
        }, inputs=['angle'], outputs=['normal'], scalars=['angle'], vectors=['normal'],
                               name='Normal', hide=True)
        tree.links.new(angle.std_out, normal.inputs['angle'])

        voronoi = make_function(tree, functions={
            "min": ["dl,normal,dot,dr,normal,dot,min,ul,normal,dot,min,ur,normal,dot,min"],
            "max": ["dl,normal,dot,dr,normal,dot,max,ul,normal,dot,max,ur,normal,dot,max"],
        }, inputs=['dl', 'dr', 'ul', 'ur', 'normal'], outputs=['min', 'max'],
                                scalars=['min', 'max'], vectors=['dl', 'dr', 'ul', 'ur', 'normal'], name='Voronoi',
                                hide=True)
        tree.links.new(normal.outputs['normal'], voronoi.inputs['normal'])
        tree.links.new(dl.std_out, voronoi.inputs['dl'])
        tree.links.new(dr.std_out, voronoi.inputs['dr'])
        tree.links.new(ur.std_out, voronoi.inputs['ur'])
        tree.links.new(ul.std_out, voronoi.inputs['ul'])

        close_to = make_function(tree, functions={
            "is": ["position,normal,dot,minimum,>,position,normal,dot,maximum,<,*"],
            "is_not": ["position,normal,dot,minimum,>,position,normal,dot,maximum,<,*,not"]
        }, name="CloseTo", inputs=['position', 'minimum', 'maximum', 'normal'], outputs=['is', 'is_not'],
                                 scalars=['minimum', 'maximum', 'is', 'is_not'],
                                 vectors=['position', 'normal'], hide=True)
        tree.links.new(position.std_out, close_to.inputs['position'])
        tree.links.new(normal.outputs['normal'], close_to.inputs['normal'])
        tree.links.new(voronoi.outputs['min'], close_to.inputs['minimum'])
        tree.links.new(voronoi.outputs['max'], close_to.inputs['maximum'])

        # projection
        projection = make_function(tree, functions={
            "location": "position,normal,normal,position,dot,projector,*,scale,sub"
        }, inputs=['position', 'projector', 'normal'], outputs=['location'], scalars=['projector'],
                                   vectors=['normal', 'position', 'location'], name="Projection", hide=True)
        tree.links.new(normal.outputs['normal'], projection.inputs['normal'])
        tree.links.new(position.std_out, projection.inputs['position'])
        tree.links.new(projector.std_out, projection.inputs['projector'])

        # inside line

        mesh = Grid(tree, size_x=20, size_y=20, vertices_y=21, vertices_x=21)
        transform_geometry = TransformGeometry(tree, translation=offset.std_out)
        delete_geometry = DeleteGeometry(tree, selection=close_to.outputs["is_not"])

        join_geometry = JoinGeometry(tree)

        # split horizontal and vertical lines
        edge_vertices = EdgeVertices(tree)
        horizontal_tester = make_function(tree, functions={
            "is": "pos1,pos2,sub,e_x,dot,abs,0,>",
        }, inputs=["pos1", "pos2"], outputs=["is", "is_not"], vectors=["pos1", "pos2"], scalars=["is"],
                                          name="HorizontalTester", hide=True)
        tree.links.new(edge_vertices.outputs["Position 1"], horizontal_tester.inputs["pos1"])
        tree.links.new(edge_vertices.outputs["Position 2"], horizontal_tester.inputs["pos2"])

        stored_attribute = StoredNamedAttribute(tree, data_type='BOOLEAN', domain='EDGE', name='horizontalAttribute',
                                                value=horizontal_tester.outputs[0])

        named_attribute = NamedAttribute(tree, data_type='BOOLEAN', name='horizontalAttribute')
        not_node = BooleanMath(tree, operation='NOT', inputs0=named_attribute.std_out, name="Not")

        # horizontal
        instance_on_edges_horizontal = InstanceOnEdges(tree, selection=named_attribute.std_out, resolution=4,
                                                       radius=0.05, name="Horizontal")
        set_pos_horizontal = SetPosition(tree, position=projection.outputs['location'])
        horizontal_mat = SetMaterial(tree, selection=None, material='plastic_joker')

        # vertical
        instance_on_edges_vertical = InstanceOnEdges(tree, selection=not_node.std_out, resolution=4, radius=0.05,
                                                     name="Vertical")
        set_pos_vertical = SetPosition(tree, position=projection.outputs['location'])
        vertical_mat = SetMaterial(tree, selection=None, material='plastic_important')

        create_geometry_line(tree, [mesh, transform_geometry, delete_geometry, stored_attribute, set_pos_horizontal,
                                    instance_on_edges_horizontal, horizontal_mat, join_geometry],
                             out=self.group_outputs.inputs[0])
        create_geometry_line(tree, [stored_attribute, set_pos_vertical, instance_on_edges_vertical, vertical_mat,
                                    join_geometry])

        # outside
        cube = CubeMesh(tree, size=[0.1] * 3)
        instance_on_points2 = InstanceOnPoints(tree, selection=close_to.outputs["is_not"], instance=cube.geometry_out)
        create_geometry_line(tree, [transform_geometry, instance_on_points2, join_geometry])

        # sphere branch
        icosphere = IcoSphere(tree, radius=0.1, subdivisions=2, name="SelectedPoints")
        instance_on_points = InstanceOnPoints(tree, instance=icosphere.geometry_out, selection=close_to.outputs["is"])
        set_pos2 = SetPosition(tree, position=projection.outputs['location'])
        cube_mat = SetMaterial(tree, material='plastic_example')
        create_geometry_line(tree, [transform_geometry, instance_on_points, set_pos2, cube_mat, join_geometry])

        # mesh line branch
        rotation = make_function(tree.nodes, functions={
            "rotation": ["pi,2,/", "0", "angle"]
        }, inputs=['angle'], outputs=['rotation'], scalars=['angle'], vectors=['rotation'], name="Rotation", hide=True)
        tree.links.new(angle.std_out, rotation.inputs['angle'])

        mesh_line = MeshLine(tree, name="ProjectionLine",
                             start_location=Vector([0, 0, -15]), end_location=Vector([0, 0, 15]))
        instance_on_edge = InstanceOnEdges(tree, radius=0.025, resolution=8)
        transform_geometry2 = TransformGeometry(tree, rotation=rotation.outputs['rotation'])
        mat_line = SetMaterial(tree, material='plastic_custom1')

        create_geometry_line(tree, [mesh_line, instance_on_edge, transform_geometry2, mat_line, join_geometry])

        # voronoi cell
        voronoi_cell = Grid(tree, name='VoronoiCell', size_x=1, size_y=1, vertices_x=2, vertices_y=2)
        transform_geometry3 = TransformGeometry(tree, translation=[0.5, 0.5, 0])
        mat_cell = SetMaterial(tree, material='plastic_drawing')
        create_geometry_line(tree, [voronoi_cell, transform_geometry3, mat_cell, join_geometry])

        # voronoi zone

        mesh_line_max = MeshLine(tree, name='SelectionZoneMax',
                                 start_location=Vector([0, 0, -15]), end_location=Vector([0, 0, 15]))
        mesh_line_min = MeshLine(tree, name='SelectionZoneMin',
                                 start_location=Vector([0, 0, -15]), end_location=Vector([0, 0, 15]))

        instance_on_edge_max = InstanceOnEdges(tree, resolution=4, radius=0.01)
        instance_on_edge_min = InstanceOnEdges(tree, resolution=4, radius=0.01)

        translation_max = make_function(tree, functions={
            "translation": "normal,maximum,scale"
        }, inputs=['normal', 'maximum'], outputs=['translation'], scalars=['maximum'],
                                        vectors=['normal', 'translation'], name='TransMax', hide=True
                                        )
        tree.links.new(normal.outputs['normal'], translation_max.inputs['normal'])
        tree.links.new(voronoi.outputs['max'], translation_max.inputs['maximum'])

        translation_min = make_function(tree, functions={
            "translation": "normal,minimum,scale"
        }, inputs=['normal', 'minimum'], outputs=['translation'], scalars=['minimum'],
                                        vectors=['normal', 'translation'], name='TransMin', hide=True
                                        )
        tree.links.new(normal.outputs['normal'], translation_min.inputs['normal'])
        tree.links.new(voronoi.outputs['min'], translation_min.inputs['minimum'])

        transform_geometry_max = TransformGeometry(tree, rotation=rotation.outputs['rotation'],
                                                   translation=translation_max.outputs['translation'])
        transform_geometry_min = TransformGeometry(tree, rotation=rotation.outputs['rotation'],
                                                   translation=translation_min.outputs['translation'])
        mat_max = SetMaterial(tree, material='plastic_drawing')
        mat_min = SetMaterial(tree, material='plastic_drawing')
        create_geometry_line(tree,
                             [mesh_line_max, instance_on_edge_max, transform_geometry_max, mat_max, join_geometry])
        create_geometry_line(tree,
                             [mesh_line_min, instance_on_edge_min, transform_geometry_min, mat_min, join_geometry])

    def get_iteration_socket(self):
        return self.repeat.repeat_input.inputs[0]


class ConvexHull2D(GeometryNodesModifier):
    def __init__(self, name='ConvexHull2D', size=10):
        self.size = size
        super().__init__(name)

    def create_node(self, tree):
        # parameters
        size = self.size

        # input nodes
        position = Position(tree)
        cube_scale = InputValue(tree, value=1, name="CubeScale")
        r_max = InputValue(tree, hide=True, name='rMax', value=1.1* size * r3)
        u = InputVector(tree,value=Vector([1,1,1]),name="Normal")
        l_max = InputValue(tree,hide=True,name='lMax', value = 0)
        p_max = InputValue(tree,hide=True,name='pMax', value = 0)
        ico_max = InputValue(tree,hide=True,name='icoMax', value = 0)
        voronoi_scale = InputValue(tree,name="voronoiScale",value = 0)
        # output nodes
        out = tree.nodes.get("Group Output")

        # grid

        mesh_line = MeshLine(tree, count=2 * size + 1, start_location=Vector([0, 0, -size]),
                             end_location=Vector([0, 0, size]))
        grid = Grid(tree, size_x=2 * size, size_y=2 * size, vertices_x=2 * size + 1, vertices_y=2 * size + 1)
        instance_on_points = InstanceOnPoints(tree, instance=grid.outputs['Mesh'])
        realize_instances = RealizeInstances(tree)
        cubie_scale = InputValue(tree, value=0.05, name="CubieScale")
        cubies = CubeMesh(tree, size=cubie_scale.std_out)
        set_material = SetMaterial(tree, material='text')
        nodes = [r_max, position]
        all_labels = ['r_max', 'position']

        selector = make_function(tree.nodes, functions={
            'selector': ['position,length,r_max,<']
        }, name='Limits', inputs=all_labels,
                                 outputs=['selector'],
                                 vectors=['position'],
                                 scalars=['r_max', 'selector'], hide=True)

        for node, label in zip(nodes, all_labels):
            tree.links.new(node.std_out, selector.inputs[label])

        create_geometry_line(tree, [cubies, set_material])

        instance_on_points2 = InstanceOnPoints(tree, selection=selector.outputs['selector'],
                                               instance=set_material.geometry_out)

        join = JoinGeometry(tree)
        shade_smooth = SetShadeSmooth(tree)
        join2 = JoinGeometry(tree)

        # voronoi cell

        cube = CubeMesh(tree,name="VoronoiCell",size=voronoi_scale.std_out)
        wire_frame= WireFrame(tree,resolution =8,radius=0.025)
        transform_geo = TransformGeometry(tree, translation=[0.5] * 3, scale=cube_scale.std_out)
        drawing = SetMaterial(tree, material='plastic_drawing', roughness=0.25, name="CubeMaterial")

        # projection line

        normalizer = make_function(tree,functions={
            'normalized':'normal,normalize',
            'neg_normalized':'normal,normalize,-1,scale',
            'u':'e_z,normal,cross,normalize,e_z,normal,cross,length,0,>,scale,e_x,normal,cross,normalize,1,e_z,normal,cross,length,0,>,-,scale,add',
            'v':'normal,e_z,normal,cross,cross,normalize,e_z,normal,cross,length,0,>,scale,normal,e_x,normal,cross,cross,normalize,1,e_z,normal,cross,length,0,>,-,scale,add',
            'euler':'e_z,normal,normalize,cross,normal,normalize,e_z,dot,acos,axis_angle_euler',
        },name="Normalizer",inputs=['normal'],outputs=['normalized','neg_normalized','u','v','euler'],
                                   vectors=['normal','normalized','neg_normalized','u','v','euler'],hide=True)
        tree.links.new(u.std_out,normalizer.inputs['normal'])

        ## length of projection line with ray-cast

        boundary_cube = CubeMesh(tree,size=[2*size]*3,name = "BoundaryCube")

        positive_ray = RayCast(tree,data_type='FLOAT_VECTOR',target_geometry=boundary_cube.geometry_out,
                               ray_direction=normalizer.outputs['normalized'],source_position=normalizer.outputs['normalized'])
        negative_ray = RayCast(tree,data_type='FLOAT_VECTOR',target_geometry=boundary_cube.geometry_out,
                               ray_direction=normalizer.outputs['neg_normalized'],source_position=normalizer.outputs['neg_normalized'])
        scaler_pos = VectorMath(tree,operation='SCALE', inputs0=positive_ray.outputs['Hit Position'],float_input=l_max.std_out,name="LineScaling")
        scaler_neg = VectorMath(tree,operation='SCALE', inputs0=negative_ray.outputs['Hit Position'],float_input=l_max.std_out,name="LineScaling")
        projection_line = MeshLine(tree, name="ProjectionLine", count=30,
                             start_location=scaler_neg.std_out, end_location=scaler_pos.std_out)
        instance_on_edge = InstanceOnEdges(tree, radius=0.0225, resolution=8)
        mat_line = SetMaterial(tree, material='plastic_custom1', name="ProjectionMaterial")

        # orthogonal plane
        # orthogonal_grid = Grid(tree,size_x=2*size,size_y=2*size,vertices_x=2*size+1,vertices_y=2*size+1,name="OrthogonalPlane")
        # rotate_orthogonal_grid = TransformGeometry(tree,rotation=normalizer.outputs['euler'])
        # orthogonal_wire_frame = WireFrame(tree,name="OrthogonalPlaneWire")

        # projector

        projector=make_function(tree,functions={
            'projection':"u,pos,u,dot,scale,v,pos,v,dot,scale,add"
        },inputs=["pos","u","v"],outputs=["projection"],vectors=["pos","u","v","projection"],
                                name="Projector",hide=True)
        tree.links.new(position.std_out,projector.inputs['pos'])
        tree.links.new(normalizer.outputs['u'],projector.inputs['u'])
        tree.links.new(normalizer.outputs['v'],projector.inputs['v'])

        # projected cube
        scale_min = make_function(tree,functions={"pos_min":"hit_pos,p_min,scale"},inputs=["p_min","hit_pos"],outputs=["pos_min"],scalars=["p_min"],vectors=["hit_pos","pos_min"],name="moveMin",hide=True)
        scale_max = make_function(tree,functions={"pos_max":"hit_pos,p_max,scale"},inputs=["p_max","hit_pos"],outputs=["pos_max"],scalars=["p_max"],vectors=["hit_pos","pos_max"],name="moveMax",hide=True)
        tree.links.new(p_max.std_out,scale_min.inputs["p_min"])
        tree.links.new(p_max.std_out,scale_max.inputs["p_max"])
        tree.links.new(positive_ray.outputs['Hit Position'],scale_max.inputs["hit_pos"])
        tree.links.new(negative_ray.outputs['Hit Position'],scale_min.inputs["hit_pos"])
        set_projected_pos = SetPosition(tree, position=projector.outputs['projection'], name="Projection")
        move_to_max = TransformGeometry(tree, translation=scale_max.outputs["pos_max"])
        wire_frame_max = WireFrame(tree, resolution=16, radius=0.025, name="WireFrameMax")
        sphere_max = IcoSphere(tree,radius=0.025,subdivisions=3)
        iop_max = InstanceOnPoints(tree,instance=sphere_max.geometry_out)
        sphere_max_mat = SetMaterial(tree,material='plastic_joker')
        convex_hull_max_material = SetMaterial(tree, material='plastic_joker', name='ConvexHullMaterialMax')
        move_to_min = TransformGeometry(tree, translation=scale_min.outputs["pos_min"])
        wire_frame_min = WireFrame(tree, resolution=16, radius=0.025, name="WireFrameMin")
        sphere_min = IcoSphere(tree, radius=0.025, subdivisions=3)
        iop_min = InstanceOnPoints(tree, instance=sphere_max.geometry_out)
        sphere_min_mat = SetMaterial(tree, material='plastic_joker')
        convex_hull_min_material = SetMaterial(tree, material='plastic_joker', name='ConvexHullMaterialMin')

        # convex hull
        convex_hull = ConvexHull(tree)
        scale_element = ScaleElements(tree,scale =1.0001) # to include boundary cases
        extrude = ExtrudeMesh(tree,mode="FACES") # give finite thickness to convex hull to make ray tracing feasible

        # select inside convex hull
        icosphere= IcoSphere(tree,subdivisions=2,name="SelectedSphere")
        selector = make_function(tree,functions={
            "is_selected":"ico_max,ray_hit,length,*,pos,length,>"
        }, inputs = ["pos","ray_hit","ico_max"],outputs=["is_selected"],vectors=["pos","ray_hit"],scalars=["is_selected","ico_max"],
                                 hide=True,name="SelectorFunction")
        tree.links.new(ico_max.std_out,selector.inputs['ico_max'])
        tree.links.new(position.std_out,selector.inputs['pos'])
        tree.links.new(positive_ray.outputs["Hit Position"],selector.inputs['ray_hit'])

        convex_hull_test = InsideConvexHull(tree,source_position=projector.outputs['projection'],
                                            ray_direction=normalizer.outputs['u'])

        and_node = BooleanMath(tree,operation="AND",inputs0=selector.outputs["is_selected"],inputs1=convex_hull_test.std_out)
        instance_on_grid_points = InstanceOnPoints(tree, selection=and_node.std_out,instance=icosphere.geometry_out)
        selected_material = SetMaterial(tree,material='plastic_example',name='SelectedMaterial')

        # create voronoi zone

        product_geometry = InstanceOnPoints(tree,instance=convex_hull.geometry_out)
        realize = RealizeInstances(tree)
        hull_hull = ConvexHull(tree)
        zone_mat = SetMaterial(tree,material="fake_glass_joker",ior=1.01,name="ZoneMaterial")

        # create geometry lines
        # lattice grid
        create_geometry_line(tree, [
            mesh_line,
            instance_on_points,
            realize_instances,
            instance_on_points2,
            join,
            shade_smooth,
            join2
        ], out=out.inputs['Geometry'])
        # voronoi cell
        create_geometry_line(tree, [cube,transform_geo,wire_frame, drawing, join2])
        # projection line
        create_geometry_line(tree, [projection_line, instance_on_edge, mat_line, join])
        # orthogonal space
        # create_geometry_line(tree,[orthogonal_grid,rotate_orthogonal_grid,orthogonal_wire_frame,join])
        # display convex hull geometry
        create_geometry_line(tree,[transform_geo,set_projected_pos,move_to_max,wire_frame_max,convex_hull_max_material,join])
        create_geometry_line(tree,[set_projected_pos,move_to_min,wire_frame_min,convex_hull_min_material,join])
        create_geometry_line(tree,[move_to_min,iop_min,sphere_min_mat,join])
        create_geometry_line(tree,[move_to_max,iop_max,sphere_max_mat,join])
        # inside convex hull test as selector
        create_geometry_line(tree,[set_projected_pos,convex_hull,scale_element,extrude,convex_hull_test])
        # selected grid points
        create_geometry_line(tree, [realize_instances,instance_on_grid_points,selected_material,join2])
        # voronoi zone
        create_geometry_line(tree,[projection_line,product_geometry,realize,hull_hull,zone_mat,join2])

class SphericalHarmonicsNode(GeometryNodesModifier):
    def __init__(self,l=0,m=0,name='SphericalHarmonics',resolution =5,**kwargs):
        """
        creates an object that represents a spherical harmonics
        :param name:
        :param l: orbital quantum number
        :param m: magnetic quantum number
        :param kwargs:
        """
        self.l=l
        self.m=m
        self.resolution=resolution
        self.kwargs = kwargs
        super().__init__(name)

    def create_node(self,tree):
        # recalculate position of vertices
        position = Position(tree,location=(-6,0))
        # convert to polar coordinates
        cart2polar = make_function(tree,name="PolarCoordinates",hide=True,location=(-5,0),
                                   functions={
                                     "r":"position,length",
                                     "theta":"position_z,acos",
                                     "phi":"position_y,position_x,atan2"
                                   },inputs=["position"],outputs=["r","theta","phi"],
                                   vectors=["position"],scalars=["r","theta","phi"])

        tree.links.new(position.std_out,cart2polar.inputs["position"])
        # create spherical harmonics terms
        y_lm = SphericalHarmonics(self.l,self.m,"theta","phi")

        real_part = ExpressionConverter(y_lm.real()).postfix()
        imag_part = ExpressionConverter(y_lm.imag()).postfix()

        if len(imag_part.strip())==0:
            imag_part="0"
        print("real: ",real_part)
        print("imag: ",imag_part)

        compute_y_lm = make_function(tree,name="Y_lm",functions={
            "re":real_part,
            "im":imag_part
        },
                                 inputs=["theta","phi"],
                                 outputs=["re","im"],scalars=["re","im","theta","phi"],hide=True,location=(-4,0))

        tree.links.new(cart2polar.outputs["theta"],compute_y_lm.inputs["theta"])
        tree.links.new(cart2polar.outputs["phi"],compute_y_lm.inputs["phi"])

        complex_analyser = make_function(tree,name="Analyser",hide=True,functions={
            "re":"re",
            "im":"im",
            "absolute":"re,re,*,im,im,*,+,sqrt",
            "phase":"im,re,atan2"
        },inputs=["re","im"],outputs=["re","im","absolute","phase"],scalars=["re","im","absolute","phase"],location=(-3,0))

        tree.links.new(compute_y_lm.outputs["re"],complex_analyser.inputs["re"])
        tree.links.new(compute_y_lm.outputs["im"],complex_analyser.inputs["im"])

        # transform position
        vals = ["re","im","absolute","phase"]
        trafo =make_function(tree,name="Transformation",hide=True,functions={
            "position":"position,re,abs,scale"
        },inputs=["position","re","im","absolute","phase"],outputs=["position"],vectors=["position"],
                             scalars=vals,location=(-2,0) )

        tree.links.new(position.std_out,trafo.inputs["position"])
        for val in vals:
            tree.links.new(complex_analyser.outputs[val],trafo.inputs[val])

        # create default spherical geometry
        sphere = UVSphere(tree,location=(-3,1),rings=2**self.resolution,segments=2**(self.resolution+1),hide=False)
        set_pos = SetPosition(tree,location=(-2,1),position=trafo.outputs["position"])

        # store the phase for coloring
        attr = StoredNamedAttribute(tree,location=(-1,-2),name="Phase",value=complex_analyser.outputs["phase"])
        material = phase2hue_material(attribute_names=["Phase"],**self.kwargs)
        self.materials.append(material)
        color = SetMaterial(tree,location=(0,-1),material=material)

        smooth = SetShadeSmooth(tree,location=(1,-1))

        create_geometry_line(tree,[sphere,set_pos,attr,color,smooth],out=self.group_outputs.inputs[0])