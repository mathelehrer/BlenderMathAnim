bl_info = {
    "name": "Platonic Solid Through Three Points",
    "blender": (3, 0, 0),
    "category": "Object",
    "author": "AI Assistant",
    "version": (1, 0, 0),
    "description": "Pick 3 objects, then create a Platonic solid whose 3 vertices pass through those points.",
}

import bpy
from mathutils import Vector, Matrix
from bpy.props import FloatVectorProperty, EnumProperty


# ---------------------------------------------------------------------------
# Geometry definitions for Platonic solids
# ---------------------------------------------------------------------------

def get_platonic_solid_data(solid_type: str):
    """
    Returns (verts, faces, ref_indices) for a given Platonic solid type.
    verts: list[tuple[float, float, float]]
    faces: list[tuple[int, ...]]  # indices into verts
    ref_indices: (i0, i1, i2)     # three vertex indices used as reference
    """
    from math import sqrt

    if solid_type == 'TETRA':
        # Regular tetrahedron using 4 vertices of a cube
        verts = [
            ( 1.0,  1.0,  1.0),
            (-1.0, -1.0,  1.0),
            (-1.0,  1.0, -1.0),
            ( 1.0, -1.0, -1.0),
        ]
        faces = [
            (0, 1, 2),
            (0, 3, 1),
            (0, 2, 3),
            (1, 3, 2),
        ]
        # Use one vertex and two edges from it as reference
        ref_indices = (0, 1, 2)
        return verts, faces, ref_indices

    if solid_type == 'CUBE':
        verts = [
            (-1.0, -1.0, -1.0),
            (-1.0, -1.0,  1.0),
            (-1.0,  1.0, -1.0),
            (-1.0,  1.0,  1.0),
            ( 1.0, -1.0, -1.0),
            ( 1.0, -1.0,  1.0),
            ( 1.0,  1.0, -1.0),
            ( 1.0,  1.0,  1.0),
        ]
        faces = [
            (0, 1, 3, 2),
            (4, 6, 7, 5),
            (0, 4, 5, 1),
            (2, 3, 7, 6),
            (0, 2, 6, 4),
            (1, 5, 7, 3),
        ]
        # Corner with two edges from it
        ref_indices = (7, 3, 5)  # (1,1,1) corner: neighbors along -Y and -Z
        return verts, faces, ref_indices

    if solid_type == 'OCTA':
        verts = [
            ( 1.0,  0.0,  0.0),
            (-1.0,  0.0,  0.0),
            ( 0.0,  1.0,  0.0),
            ( 0.0, -1.0,  0.0),
            ( 0.0,  0.0,  1.0),
            ( 0.0,  0.0, -1.0),
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
        ref_indices = (0, 2, 4)  # three mutually adjacent vertices
        return verts, faces, ref_indices

    if solid_type == 'ICOSA':
        # Icosahedron
        phi = (1.0 + sqrt(5.0)) / 2.0
        a = 1.0
        verts = [
            ( 0,  a,  phi),
            ( 0, -a,  phi),
            ( 0,  a, -phi),
            ( 0, -a, -phi),
            ( a,  phi, 0),
            (-a,  phi, 0),
            ( a, -phi, 0),
            (-a, -phi, 0),
            ( phi, 0,  a),
            (-phi, 0,  a),
            ( phi, 0, -a),
            (-phi, 0, -a),
        ]
        faces = [
            (0, 1, 8),  (0, 8, 4),  (0, 4, 5),  (0, 5, 9),  (0, 9, 1),
            (1, 6, 8),  (1, 7, 6),  (1, 9, 7),  (2, 3, 10), (2, 11, 3),
            (2, 5, 11), (2, 4, 5),  (2, 10, 4), (3, 7, 9),  (3, 11, 7),
            (3, 6, 7),  (3, 10, 6), (4, 10, 8), (5, 4, 2),  (5, 2, 11),
            (6, 10, 8), (7, 9, 1),  (8, 6, 10), (9, 5, 0),  (11, 5, 2),
        ]
        # Choose a vertex and two of its neighbors as reference
        ref_indices = (0, 1, 8)  # triangle from first face
        return verts, faces, ref_indices

    if solid_type == 'DODECA':
        # Dodecahedron dual of icosahedron
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

        # Faces defined by vertex indices
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
            (8, 1, 9, 4, 16),  # some faces repeated for completeness
        ]
        # Use three vertices from one pentagon
        ref_indices = faces[0][:3]
        return verts, faces, ref_indices

    raise ValueError(f"Unknown solid_type: {solid_type}")


# ---------------------------------------------------------------------------
# Math: Similarity transform from 3 canonical vertices to 3 target points
# ---------------------------------------------------------------------------

def compute_similarity_transform(a0: Vector, a1: Vector, a2: Vector,
                                 p0: Vector, p1: Vector, p2: Vector):
    """
    Compute (scale, rotation_matrix, translation) such that:
        p_i ≈ R * (s * a_i) + t  for i = 0,1,2
    using a0 as origin and a1,a2 as edges.
    """
    # Edge vectors in canonical space
    A1 = a1 - a0
    A2 = a2 - a0

    # Edge vectors in target space
    P1 = p1 - p0
    P2 = p2 - p0

    if A1.length == 0 or A2.length == 0:
        raise ValueError("Canonical reference vertices are degenerate")

    if P1.length == 0 or P2.length == 0:
        raise ValueError("Target points are degenerate / too close")

    # Single uniform scale from edge length ratio
    scale = P1.length / A1.length

    # Build orthonormal bases in canonical and target spaces
    def make_basis(e1, e2):
        u = e1.normalized()
        # component of e2 orthogonal to u
        v = (e2 - e2.project(u))
        if v.length == 0:
            raise ValueError("Reference edges are collinear")
        v.normalize()
        w = u.cross(v)
        return u, v, w

    uA, vA, wA = make_basis(A1, A2)
    uP, vP, wP = make_basis(P1, P2)

    # Columns are basis vectors
    BA = Matrix((uA, vA, wA)).transposed()  # 3x3
    BP = Matrix((uP, vP, wP)).transposed()  # 3x3

    # Rotation that maps canonical basis to target basis
    R = BP @ BA.inverted()

    # Translation so that a0 maps to p0
    t = p0 - R @ (scale * a0)

    return scale, R, t


def apply_similarity_to_vertices(verts, scale, R: Matrix, t: Vector):
    """Return transformed vertices as list of tuples."""
    out = []
    for v in verts:
        v_world = R @ (scale * Vector(v)) + t
        out.append((v_world.x, v_world.y, v_world.z))
    return out


# ---------------------------------------------------------------------------
# Operator: After having 3 points, choose solid & create mesh
# ---------------------------------------------------------------------------

class OBJECT_OT_add_platonic_through_points(bpy.types.Operator):
    """Create a Platonic solid whose 3 vertices pass through 3 given points"""
    bl_idname = "object.add_platonic_through_points"
    bl_label = "Add Platonic Solid Through Points"
    bl_options = {'REGISTER', 'UNDO'}

    # These are passed in from the picker operator
    p0: FloatVectorProperty(size=3)
    p1: FloatVectorProperty(size=3)
    p2: FloatVectorProperty(size=3)

    solid_type: EnumProperty(
        name="Solid",
        items=[
            ('TETRA', "Tetrahedron", ""),
            ('CUBE', "Cube", ""),
            ('OCTA', "Octahedron", ""),
            ('DODECA', "Dodecahedron", ""),
            ('ICOSA', "Icosahedron", ""),
        ],
        default='TETRA',
    )

    def invoke(self, context, event):
        # Show a dialog so user can choose the solid
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
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

        # Create mesh and object
        mesh = bpy.data.meshes.new(f"{self.solid_type}_ThroughPoints_Mesh")
        mesh.from_pydata(verts_world, [], faces)
        mesh.update()

        obj = bpy.data.objects.new(f"{self.solid_type}_ThroughPoints", mesh)
        context.scene.collection.objects.link(obj)
        context.view_layer.objects.active = obj
        obj.select_set(True)

        self.report({'INFO'}, f"{self.solid_type} created through 3 points.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Operator: Pick three objects in viewport
# ---------------------------------------------------------------------------

class OBJECT_OT_pick_three_points_for_platonic(bpy.types.Operator):
    """Pick 3 objects by clicking in the viewport; then choose a Platonic solid"""
    bl_idname = "object.pick_three_points_for_platonic"
    bl_label = "Pick 3 Objects for Platonic Solid"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "Go to a 3D View to use this operator")
            return {'CANCELLED'}

        self.picked_names = []
        self.report({'INFO'}, "Click objects to pick them (3 needed). ESC to cancel.")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        # Cancel with ESC
        if event.type == 'ESC':
            self.report({'INFO'}, "Cancelled picking.")
            return {'CANCELLED'}

        # Detect selection on LEFTMOUSE release
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            obj = context.active_object
            if obj and obj.name not in self.picked_names:
                self.picked_names.append(obj.name)
                self.report({'INFO'},
                            f"Picked: {obj.name} ({len(self.picked_names)}/3)")

            # When we have 3 picks -> call the solid-creation operator
            if len(self.picked_names) == 3:
                o0 = bpy.data.objects.get(self.picked_names[0])
                o1 = bpy.data.objects.get(self.picked_names[1])
                o2 = bpy.data.objects.get(self.picked_names[2])

                if not (o0 and o1 and o2):
                    self.report({'ERROR'}, "One or more picked objects no longer exist.")
                    return {'CANCELLED'}

                # Call the next operator, which will ask which solid to create
                bpy.ops.object.add_platonic_through_points(
                    'INVOKE_DEFAULT',
                    p0=o0.location,
                    p1=o1.location,
                    p2=o2.location,
                )

                return {'FINISHED'}

        # Let Blender handle other events normally
        return {'PASS_THROUGH'}


# ---------------------------------------------------------------------------
# UI Panel
# ---------------------------------------------------------------------------

class VIEW3D_PT_platonic_through_points(bpy.types.Panel):
    """Panel in the 3D Viewport"""
    bl_label = "Platonic Through 3 Points"
    bl_idname = "VIEW3D_PT_platonic_through_points"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Platonic"

    def draw(self, context):
        layout = self.layout
        layout.operator(
            OBJECT_OT_pick_three_points_for_platonic.bl_idname,
            text="Pick 3 Objects & Create"
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    OBJECT_OT_add_platonic_through_points,
    OBJECT_OT_pick_three_points_for_platonic,
    VIEW3D_PT_platonic_through_points,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()