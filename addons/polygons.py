import itertools

import numpy as np

pi = np.pi
tau = 2 * pi

bl_info = {
    "name": "Regular Polygon from two selected Objects",
    "author": "NumberCruncher",
    "version": (1, 1, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Regular Polygons",
    "category": "Object",
    "description": "Create a regular Polygon matching the centers of the two selected objects",
}

import bpy
from mathutils import Vector, Matrix
from bpy.props import (
    FloatVectorProperty,
    FloatProperty,
    IntProperty,
    EnumProperty,
    BoolProperty,
    CollectionProperty,
)

# ------------------------------------------------------------------------
# Polygon data
# ------------------------------------------------------------------------

def get_polygon_data(n: int):

    # --- regular polygon
    r = 1.0 /2/ np.sin(np.pi / n)

    verts = []
    # ring (0..n-1)
    for i in range(n):
        ang = 2.0 * np.pi * i / n
        verts.append((r * np.cos(ang), r * np.sin(ang),0))

    # Faces
    bottom = tuple(range(0,n))
    faces = [bottom]

    return verts, faces

# internal codes for the polygons
POLYGONS = {
    3:'TRIANGLE',
    4:'SQUARE',
    5:'PENTAGON',
    6:'HEXAGON',
    7:'HEPTAGON',
    8:'OCTAGON',
    9:'NONAGON',
    10:'DECAGON',
    11:'POLY11',
    12:'POLY12',
    13:'POLY13',
    14:'POLY14',
    15:'POLY15',
}
# Colors for each regular polygon type (RGBA)
POLYGON_COLORS = {
    'HEPTAGON': (1.0, 0.2, 0.2, 1.0),   # Red
    'SQUARE':  (0.5, 0.5, 0.5, 1.0),   # gray
    'PENTAGON':  (10/255, 158/255, 115/255, 1.0),   # joker
    'HEXAGON':(0, 0.25, 0.5, 1.0),  # match with truncated octahedron
    'TRIANGLE': (0., 1.0, 0., 1.0),   # Green
    'OCTAGON': (0.8, 0.5, 0.2, 1.0),
    'NONAGON': (0.4, 0.2, 0.8, 1.0),
    'DECAGON': (1,0,1, 1.0),# match with truncated icosidodecahedron
    'POLY11': (0.2, 0.8, 0.5, 1.0),
    'POLY12': (0.5, 0.2, 0.8, 1.0),
    'POLY13': (0.8, 0.2, 0.5, 1.0),
    'POLY14': (0.5, 0.8, 0.2, 1.0),
    'POLY15': (0.2, 0.5, 0.8, 1.0)
}

# edge material
def get_edge_material():
    mat = bpy.data.materials.get("EdgeMaterial")
    if mat is not None:
        return mat
    mat=bpy.data.materials.new("EdgeMaterial")
    mat.use_nodes=True
    tree = mat.node_tree
    nodes = tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value=[240/255, 228/255, 66/255,1]
    return mat


# ------------------------------------------------------------------------
# Operator: Create regular polygon from stored points
# ------------------------------------------------------------------------

def polygon_items(self, context):
    all_items = [
        ('TRIANGLE', "Triange (60°)", ""),
        ('SQUARE', "Square (90°)", ""),
        ('PENTAGON', "Pentagon (108°)", ""),
        ('HEXAGON', "Hexagon (120°)", ""),
        ('HEPTAGON', "7gon (ca. 128.6°)", ""),
        ('OCTAGON', "8gon (135°)", ""),
        ('NONAGON', "9gon (140°)", ""),
        ('DECAGON', "10gon (144°)", ""),
        ('POLY11',"11gon (147.3°)",""),
        ('POLY12',"12gon (147.3°)",""),
        ('POLY13',"13gon (150°)",""),
        ('POLY14',"14gon (152.3°)",""),
        ('POLY15',"15gon (156°)",""),
    ]

    return [
        item for item in all_items
    ]

class PointItem(bpy.types.PropertyGroup):
    """Helper property group to store variable number of points in the operator"""
    vec: FloatVectorProperty(name="Location", size=3)

class OBJECT_OT_remove_spheres(bpy.types.Operator):
    """Remove spheres from selected objects"""
    bl_idname = "object.remove_spheres"
    bl_label = "Remove Spheres"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if "Sphere_Vert" in obj.name:
                obj.select_set(True)

        bpy.ops.object.delete()
        return {'FINISHED'}

class OBJECT_OT_add_polygon_from_selection(bpy.types.Operator):
    """Create a polygon fitted through selected objects two objects"""
    bl_idname = "object.add_polygon_from_selection"
    bl_label = "Add Regular Polygon From Selection"
    bl_options = {'REGISTER', 'UNDO'}

    # We use a CollectionProperty to store dynamic points
    points: CollectionProperty(type=PointItem)

    # Keep these for reference or logic
    point_count: IntProperty(default=0)
    face_sides: IntProperty(default=3)

    ngon_sides: IntProperty(
        name="Sides",
        description="Number of sides for the regular polygon",
        default = 3,
        min =3 ,
        max = 60
    )

    flip_face: BoolProperty(
        name="Flip",
        description="Flip the orientation of the polygon",
        default=False
    )

    vertex_sphere_radius: FloatProperty(
        name="Vertex Sphere Radius",
        description="Radius of spheres to place at each vertex",
        default=0.1,
        min=0.001
    )

    edge_radius: FloatProperty(
        name="Edge Radius",
        description ="Radius of the cylinder to place at each edge",
        default = 0.05,
        min = 0.001
    )

    def invoke(self, context, event):
        # Immediately grab selected objects
        objs = context.selected_objects
        if len(objs) != 2:
            self.report({'ERROR'}, "Please select exactly 2 objects.")
            return {'CANCELLED'}

        self.points.clear()
        for obj in objs:
            item = self.points.add()
            item.vec = obj.location.copy()

        self.point_count = len(objs)

        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "ngon_sides")
        layout.prop(self, "flip_face")
        layout.prop(self, "vertex_sphere_radius")
        layout.prop(self, "edge_radius")
        # layout.separator()
        # layout.label(text=f"Selected {self.point_count} locations:")
        # box = layout.box()
        # for i, item in enumerate(self.points):
        #     box.prop(item, "vec", text=f"Object {i+1}")

    # geometry modifier
    def add_edge_modifier(self, obj):
        modifier = obj.modifiers.new(type="NODES", name="Edges")
        tree = bpy.data.node_groups.new(name="EdgeModifier", type="GeometryNodeTree")
        modifier.node_group = tree

        nodes = tree.nodes
        links = tree.links

        m2c = nodes.new(type="GeometryNodeMeshToCurve")
        m2c.location = (-600, 0)
        ins = nodes.new("NodeGroupInput")
        tree.interface.new_socket(name="Mesh", in_out="INPUT", socket_type="NodeSocketGeometry")
        out = nodes.new("NodeGroupOutput")
        tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

        ins.location = (-800, 0)
        links.new(ins.outputs[0], m2c.inputs["Mesh"])

        c2m = nodes.new(type="GeometryNodeCurveToMesh")
        c2m.location = (-400, 0)
        links.new(m2c.outputs["Curve"], c2m.inputs["Curve"])

        circ = nodes.new(type="GeometryNodeCurvePrimitiveCircle")
        circ.location = (-600, -200)
        circ.inputs["Resolution"].default_value = 8
        circ.inputs["Radius"].default_value = self.edge_radius
        links.new(circ.outputs["Curve"], c2m.inputs["Profile Curve"])

        mat = nodes.new(type="GeometryNodeSetMaterial")
        mat.location = (-200, 0)
        mat.inputs["Material"].default_value = get_edge_material()
        links.new(c2m.outputs["Mesh"], mat.inputs["Geometry"])

        join = nodes.new(type="GeometryNodeJoinGeometry")
        join.location = (0, 0)
        links.new(ins.outputs[0], join.inputs["Geometry"])
        links.new(mat.outputs[0], join.inputs["Geometry"])

        out.location = (200, 0)
        links.new(join.outputs["Geometry"], out.inputs[0])

    def execute(self, context):
        if len(self.points) < 2:
            return {'CANCELLED'}

        # No complicated transformations are required in the two-dimensional case.
        # Just compute the center of the polygon and the initial start angle to have vertices algined with existing edge

        p0 = Vector(self.points[0].vec)
        p1 = Vector(self.points[1].vec)
        Origin = Vector()

        edge_vec = p1-p0
        vec_xy = Vector((edge_vec.x,edge_vec.y,0))
        l=vec_xy.length

        if l<1e-6:
            self.report({'ERROR'}, "Selected points are too close to each other.")
            return {'CANCELLED'}

        n = self.ngon_sides

        # Properties of regular n-gon with side l
        # d_c: Distance from Center of Polygon to Edge Midpoint (Apothem)
        d_c = l / (2.0 * np.tan(np.pi / n))

        # Direction from M to Center (perpendicular to edge in XY plane)
        # vec_xy = (dx, dy, 0) -> perp = (-dy, dx, 0) which is 90 deg rotation CCW
        perp = Vector((-vec_xy.y, vec_xy.x, 0.0)).normalized()

        # Determine rotation direction based on flip
        # Standard (CCW edge P0->P1): Center is to the left (perp)
        if self.flip_face:
            perp = -perp
            angle_step = -tau / n
        else:
            angle_step = tau / n

        m = (p0 + p1) / 2
        center = m+perp*d_c
        start_angle = np.arctan2(p0.y-center.y,p0.x-center.x)

        verts_world=[]
        r = l/(2*np.sin(pi/n))
        for i in range(n):
            theta = start_angle + i * angle_step
            vx = center.x + r * np.cos(theta)
            vy = center.y + r * np.sin(theta)
            verts_world.append(Vector((vx, vy, 0)))

        mesh_name = f"{n}gon__Mesh"
        obj_name = POLYGONS.get(n,"POLY_"+str(n))

        faces = [tuple(range(n))]
        mesh = bpy.data.meshes.new(mesh_name)
        mesh.from_pydata(verts_world, [], faces)
        mesh.update()

        obj = bpy.data.objects.new(obj_name, mesh)
        self.add_edge_modifier(obj)
        context.scene.collection.objects.link(obj)


        # --- Add Material ---
        mat_name = f"Mat_{obj_name}"
        mat = bpy.data.materials.get(mat_name)
        if not mat:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get('Principled BSDF')
            if bsdf:
                color = POLYGON_COLORS.get(POLYGONS.get(n,"POLY_"+str(n)),(1.,1.,1.,1.))
                bsdf.inputs['Base Color'].default_value = color
                bsdf.inputs["Roughness"].default_value = 0.1

        # --- Add Material ---
        mat_name = "Mat_Sphere"
        sphere_mat = bpy.data.materials.get(mat_name)
        if not sphere_mat:
            sphere_mat = bpy.data.materials.new(name=mat_name)
            sphere_mat.use_nodes = True
            bsdf = sphere_mat.node_tree.nodes.get('Principled BSDF')
            if bsdf:
                color = (0.3,0,0,1.0)
                bsdf.inputs['Base Color'].default_value = color
                bsdf.inputs["Roughness"].default_value = 0.1

        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        # --- Add Spheres at Vertices ---
        if self.vertex_sphere_radius > 0:
            # Create a shared mesh for all spheres to be efficient
            bpy.ops.mesh.primitive_ico_sphere_add(
                radius=self.vertex_sphere_radius,subdivisions=1
            )
            sphere_obj = context.active_object
            sphere_mesh = sphere_obj.data
            # We don't need the initial object, just the mesh data
            bpy.data.objects.remove(sphere_obj)

            sphere_collection_name = f"{n}gon_VertexSpheres"
            sphere_collection = bpy.data.collections.new(sphere_collection_name)
            context.scene.collection.children.link(sphere_collection)

            for i, v_co in enumerate(verts_world):
                v_sphere = bpy.data.objects.new(f"Sphere_Vert_{i}", sphere_mesh)
                v_sphere.location = v_co
                sphere_collection.objects.link(v_sphere)

                # Optional: parent spheres to the polygon
                v_sphere.parent = obj
                # Apply same material
                v_sphere.data.materials.append(sphere_mat)

        # Select the new object
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj

        self.report(
            {'INFO'},
            f"{POLYGONS.get(n,'POLY_'+str(n))} created from selection."
        )
        return {'FINISHED'}


# ------------------------------------------------------------------------
# UI Panel
# ------------------------------------------------------------------------

class VIEW3D_PT_Polygon_from_selection(bpy.types.Panel):
    """Panel in the 3D Viewport"""
    bl_label = "Regular Polygon From Selection"
    bl_idname = "VIEW3D_PT_polygon_from_selection"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Regular Polygons"

    def draw(self, context):
        layout = self.layout

        # Check selection count
        n = len(context.selected_objects)
        if n==2:
            layout.operator(
                OBJECT_OT_add_polygon_from_selection.bl_idname,
                text=f"Create from 2 Objects"
            )
        else:
            layout.label(text="Select  objects", icon='INFO')

        layout.operator(OBJECT_OT_remove_spheres.bl_idname, text="Remove Spheres")


# ------------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------------

classes = (
    PointItem,
    OBJECT_OT_add_polygon_from_selection,
    OBJECT_OT_remove_spheres,
    VIEW3D_PT_Polygon_from_selection,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()