# platonic_from_points_picker.py
bl_info = {
    "name": "Platonic From 3–5 Objects Picker",
    "author": "NumberCruncher",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Platonic",
    "category": "Object",
    "description": "Pick 3–5 objects and create a Platonic solid matching the n-gon count",
}

import bpy
from mathutils import Vector, Matrix
from bpy.props import (
    FloatVectorProperty,
    IntProperty,
    EnumProperty,
)


# ------------------------------------------------------------------------
# Platonic data
# ------------------------------------------------------------------------

def get_platonic_solid_data(solid_type: str):
    """
    Returns (verts, faces, ref_indices) for a given Platonic solid type.
    verts: list[tuple[float, float, float]]
    faces: list[tuple[int, ...]]  # indices into verts
    ref_indices: (i0, i1, i2)     # three vertex indices used as reference
    """
    from math import sqrt

    if solid_type == 'TETRA':
        verts = [
            (1.0,  1.0,  1.0),
            (-1.0, -1.0,  1.0),
            (-1.0,  1.0, -1.0),
            (1.0, -1.0, -1.0),
        ]
        faces = [
            (0, 1, 2),
            (0, 3, 1),
            (0, 2, 3),
            (1, 3, 2),
        ]
        ref_indices = (0, 1, 2)
        return verts, faces, ref_indices

    if solid_type == 'CUBE':
        verts = [
            (-1.0, -1.0, -1.0),
            (-1.0, -1.0,  1.0),
            (-1.0,  1.0, -1.0),
            (-1.0,  1.0,  1.0),
            (1.0, -1.0, -1.0),
            (1.0, -1.0,  1.0),
            (1.0,  1.0, -1.0),
            (1.0,  1.0,  1.0),
        ]
        faces = [
            (0, 1, 3, 2),
            (4, 6, 7, 5),
            (0, 4, 5, 1),
            (2, 3, 7, 6),
            (0, 2, 6, 4),
            (1, 5, 7, 3),
        ]
        ref_indices = (7, 3, 5)
        return verts, faces, ref_indices

    if solid_type == 'OCTA':
        verts = [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ]
        faces = [
            (0, 2, 4),
            (2, 1, 4),
            (1, 3, 4),
            (3, 0, 4),
            (2, 0, 5),
            (1, 2, 5),
            (3, 1, 5),
            (0, 3, 5),
        ]
        ref_indices = (0, 2, 4)
        return verts, faces, ref_indices

    if solid_type == 'ICOSA':
        phi = (1.0 + sqrt(5.0)) / 2.0
        a = 1.0
        verts = [
            (0,  a,  phi),
            (0, -a,  phi),
            (0,  a, -phi),
            (0, -a, -phi),
            (a,  phi, 0),
            (-a, phi, 0),
            (a, -phi, 0),
            (-a, -phi, 0),
            (phi, 0,  a),
            (-phi, 0,  a),
            (phi, 0, -a),
            (-phi, 0, -a),
        ]
        faces = [
            (0, 1, 8),  (0, 8, 4),  (0, 4, 5),  (0, 5, 9),  (0, 9, 1),
            (1, 6, 8),  (1, 7, 6),  (1, 9, 7),  (2, 3, 10), (2, 11, 3),
            (2, 5, 11), (2, 4, 5),  (2, 10, 4), (3, 7, 9),  (3, 11, 7),
            (3, 6, 7),  (3, 10, 6), (4, 10, 8), (5, 4, 2),  (5, 2, 11),
            (6, 10, 8), (7, 9, 1),  (8, 6, 10), (9, 5, 0),  (11, 5, 2),
        ]
        ref_indices = (0, 1, 8)
        return verts, faces, ref_indices

    if solid_type == 'DODECA':
        phi = (1.0 + sqrt(5.0)) / 2.0
        inv_phi = 1.0 / phi

        verts = []
        # (±1, ±1, ±1)
        for x in (-1, 1):
            for y in (-1, 1):
                for z in (-1, 1):
                    verts.append((x, y, z))

        # (0, ±1/φ, ±φ) and permutations
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                verts.append((0.0, s1 * inv_phi, s2 * phi))
                verts.append((s1 * inv_phi, s2 * phi, 0.0))
                verts.append((s1 * phi, 0.0, s2 * inv_phi))

        faces = [
            (0, 8, 9, 4, 16),
            (0, 16, 2, 10, 12),
            (0, 12, 13, 1, 8),
            (1, 13, 15, 5, 17),
            (1, 17, 3, 11, 9),
            (1, 9, 8, 0, 13),
            (2, 14, 6, 18, 10),
            (2, 16, 4, 20, 14),
            (3, 19, 7, 15, 13),
            (3, 17, 5, 21, 19),
            (4, 9, 11, 6, 20),
            (5, 15, 7, 22, 21),
            (6, 11, 3, 19, 18),
            (7, 19, 21, 22, 23),
            (8, 1, 9, 4, 16),
        ]
        ref_indices = faces[0][:3]
        return verts, faces, ref_indices

    raise ValueError(f"Unknown solid_type: {solid_type}")


# Face-sidedness for each solid
SOLID_FACE_SIDES = {
    'TETRA': 3,
    'OCTA': 3,
    'ICOSA': 3,
    'CUBE': 4,
    'DODECA': 5,
}


# ------------------------------------------------------------------------
# Similarity transform (fit solid so 3 canonical vertices match 3 points)
# ------------------------------------------------------------------------

def compute_similarity_transform(a0, a1, a2, p0, p1, p2):
    """
    Find scale s, rotation R, translation t so that
    R * (a_i * s) + t ~= p_i for i=0,1,2.
    """
    vA1 = a1 - a0
    vA2 = a2 - a0
    vP1 = p1 - p0
    vP2 = p2 - p0

    lenA1 = vA1.length
    lenP1 = vP1.length
    if lenA1 < 1e-8 or lenP1 < 1e-8:
        raise ValueError("Degenerate reference configuration")

    scale = lenP1 / lenA1

    uA = vA1.normalized()
    wA = vA1.cross(vA2)
    if wA.length < 1e-8:
        raise ValueError("Canonical reference points are collinear")
    wA.normalize()
    vA = wA.cross(uA)

    uP = vP1.normalized()
    wP = vP1.cross(vP2)
    if wP.length < 1e-8:
        raise ValueError("Target points are collinear")
    wP.normalize()
    vP = wP.cross(uP)

    RA = Matrix((uA, vA, wA)).transposed()
    RP = Matrix((uP, vP, wP)).transposed()

    R = RP @ RA.transposed()
    R.resize_4x4()

    a0_scaled = a0 * scale
    t = p0 - (R.to_3x3() @ a0_scaled)

    return scale, R, t


def apply_similarity_to_vertices(verts, scale, R, t):
    result = []
    for v in verts:
        v_vec = Vector(v) * scale
        v_world = R.to_3x3() @ v_vec + t
        result.append(v_world)
    return result


# ------------------------------------------------------------------------
# Operator: Create Platonic solid from stored points
# ------------------------------------------------------------------------

def platonic_items_for_ngon(self, context):
    face_sides = getattr(self, "face_sides", 3)
    all_items = [
        ('TETRA', "Tetrahedron (triangles)", ""),
        ('OCTA', "Octahedron (triangles)", ""),
        ('ICOSA', "Icosahedron (triangles)", ""),
        ('CUBE', "Cube (quads)", ""),
        ('DODECA', "Dodecahedron (pentagons)", ""),
    ]
    return [
        item for item in all_items
        if SOLID_FACE_SIDES[item[0]] == face_sides
    ]


class OBJECT_OT_add_platonic_from_points(bpy.types.Operator):
    """Create a Platonic solid fitted through 3 picked points"""
    bl_idname = "object.add_platonic_from_points"
    bl_label = "Add Platonic Solid From Points"
    bl_options = {'REGISTER', 'UNDO'}

    # p0: FloatVectorProperty(size=3)
    # p1: FloatVectorProperty(size=3)
    # p2: FloatVectorProperty(size=3)
    # p3: FloatVectorProperty(size=3)
    # p4: FloatVectorProperty(size=3)



    point_count: IntProperty(min=3, max=5, default=3)
    face_sides: IntProperty(min=3, max=5, default=3)

    solid_type: EnumProperty(
        name="Solid",
        description="Platonic solid with faces that are n-gons with n = number of picked points",
        items=platonic_items_for_ngon,
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        # compute the face center
        p0 = Vector(self.p0)
        p1 = Vector(self.p1)
        p2 = Vector(self.p2)
        center = (p0 + p1 + p2) / 3.0

        # Use first 3 points to define the transform
        p0 = Vector(self.p0)
        p1 = Vector(self.p1)
        p2 = Vector(self.p2)

        try:
            verts_canon, faces, ref_indices = get_platonic_solid_data(self.solid_type)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        a0 = Vector(verts_canon[ref_indices[0]])
        a1 = Vector(verts_canon[ref_indices[1]])
        a2 = Vector(verts_canon[ref_indices[2]])

        try:
            scale, R, t = compute_similarity_transform(a0, a1, a2, p0, p1, p2)
        except ValueError as e:
            self.report({'ERROR'}, f"Cannot fit solid to points: {e}")
            return {'CANCELLED'}

        verts_world = apply_similarity_to_vertices(verts_canon, scale, R, t)

        mesh = bpy.data.meshes.new(f"{self.solid_type}_FromPoints_Mesh")
        mesh.from_pydata(verts_world, [], faces)
        mesh.update()

        obj = bpy.data.objects.new(f"{self.solid_type}_FromPoints", mesh)
        context.scene.collection.objects.link(obj)
        context.view_layer.objects.active = obj
        obj.select_set(True)

        self.report(
            {'INFO'},
            f"{self.solid_type} (faces: {self.face_sides}-gons) created from {self.point_count} points."
        )
        return {'FINISHED'}


# ------------------------------------------------------------------------
# Operator: pick 3–5 objects in the viewport
# ------------------------------------------------------------------------

class OBJECT_OT_pick_points_for_platonic(bpy.types.Operator):
    """Pick 3–5 objects in the viewport, then choose a matching Platonic solid"""
    bl_idname = "object.pick_points_for_platonic"
    bl_label = "Pick 3–5 Objects for Platonic Solid"
    bl_options = {'REGISTER', 'UNDO'}

    picked_names: list[str] | None = None

    def invoke(self, context, event):
        self.picked_names = []
        if 3<= len(context.selected_objects) <=6:
            for obj in context.selected_objects:
                self.picked_names.append(obj.name)
            self.report({'INFO'},"Create solid from selected objects: "+str(self.picked_names))
            return self.finish_and_spawn_operator(context)

        if context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "Go to a 3D View to use this operator")
            return {'CANCELLED'}


        self.report(
            {'INFO'},
            "Click objects to pick them (3–5 needed). ESC to cancel."
        )
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            self.report({'INFO'}, "Cancelled picking.")
            return {'CANCELLED'}

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            obj = context.active_object
            if obj and obj.name not in self.picked_names:
                self.picked_names.append(obj.name)
                self.report(
                    {'INFO'},
                    f"Picked: {obj.name} ({len(self.picked_names)}/3–5)"
                )

            if len(self.picked_names) >= 3:
                # Stop once we reach 5, or user can press ENTER to finish early.
                if len(self.picked_names) == 5:
                    return self.finish_and_spawn_operator(context)

        if event.type in {'RET', 'NUMPAD_ENTER'}:
            if self.picked_names and 3 <= len(self.picked_names) <= 5:
                return self.finish_and_spawn_operator(context)
            else:
                self.report({'WARNING'}, "Need between 3 and 5 picks to continue.")
                return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    def finish_and_spawn_operator(self, context):

        n = len(self.picked_names)
        self.report({'INFO'}, f"Selected {n} objects.")
        if n < 3 or n > 5:
            self.report({'ERROR'}, "Need between 3 and 5 objects.")
            return {'CANCELLED'}

        mapping = {
            3: 3,
            4: 4,
            5: 5,
        }
        face_sides = mapping.get(n)
        if face_sides is None:
            self.report({'ERROR'}, "No Platonic solids for this number of points.")
            return {'CANCELLED'}

        objs = []
        for name in self.picked_names:
            obj = bpy.data.objects.get(name)
            if not obj:
                self.report({'ERROR'}, f"Object '{name}' no longer exists.")
                return {'CANCELLED'}
            objs.append(obj)

        def loc(i):
            return objs[i].location.copy() if i < len(objs) else objs[-1].location.copy()

        points=[obj.location.copy() for obj in objs]

        # create keyword arguments for operator
        kwargs = {}
        for i in range(len(points)):
            kwargs[f"p{i}"] = points[i]
        kwargs["point_count"] = len(points)
        kwargs["face_sides"] = face_sides

        bpy.ops.object.add_platonic_from_points(
            'INVOKE_DEFAULT',
            **kwargs
        )

        return {'FINISHED'}


# ------------------------------------------------------------------------
# UI Panel
# ------------------------------------------------------------------------

class VIEW3D_PT_platonic_from_points(bpy.types.Panel):
    """Panel in the 3D Viewport"""
    bl_label = "Platonic From 3–5 Points"
    bl_idname = "VIEW3D_PT_platonic_from_points"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Platonic"

    def draw(self, context):
        layout = self.layout
        layout.operator(
            OBJECT_OT_pick_points_for_platonic.bl_idname,
            text="Select Objects and hit Enter & Esc to cancel"
        )


# ------------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------------

classes = (
    OBJECT_OT_add_platonic_from_points,
    OBJECT_OT_pick_points_for_platonic,
    VIEW3D_PT_platonic_from_points,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()