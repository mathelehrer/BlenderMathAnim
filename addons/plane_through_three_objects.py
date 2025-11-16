# plane_from_three_objects_addon.py
bl_info = {
    "name": "Plane from Three Objects Picker",
    "blender": (3, 0, 0),
    "category": "Object",
    "author": "NumberCruncher",
    "version": (1, 0, 0),
    "description": "Pick three objects with the mouse and create a plane through them",
}

import bpy
from mathutils import Vector, Matrix


def create_plane_from_points(p0: Vector, p1: Vector, p2: Vector):
    """
    Create a plane defined by p0, p1, p2.
    Returns the created object, or None if the points are collinear/invalid.
    """

    v1 = p1 - p0
    v2 = p2 - p0

    if v1.length == 0 or v2.length == 0:
        return None

    n = v1.cross(v2)
    if n.length < 1e-6:
        return None

    n.normalize()
    u = v1.normalized()
    v = n.cross(u).normalized()

    center = (p0 + p1 + p2) / 3;
    M = Matrix((
        (u.x, v.x, n.x, center.x),
        (u.y, v.y, n.y, center.y),
        (u.z, v.z, n.z, center.z),
        (0, 0, 0, 1)
    ))

    sx = (center - p0).length
    sy = (center - p1).length
    sz = (center - p2).length

    scale = max(sx, sy, sz)

    bpy.ops.mesh.primitive_plane_add()
    plane = bpy.context.active_object
    plane.matrix_world = M
    plane.scale = [scale] * 3
    plane.name="PlaneFromThreePoints"
    bpy.context.scene.collection.objects.link(plane)

    return plane



class OBJECT_OT_pick_three_objects(bpy.types.Operator):
    """Pick 3 objects by clicking in the viewport; creates a plane through them"""
    bl_idname = "object.pick_three_objects"
    bl_label = "Pick 3 Objects for Plane"
    bl_options = {'REGISTER', 'UNDO'}

    picked_names: list[str] | None = None

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
                self.report({'INFO'}, f"Picked: {obj.name} "
                                      f"({len(self.picked_names)}/3)")

            # After 3 picks → create plane
            if len(self.picked_names) == 3:
                o0 = bpy.data.objects.get(self.picked_names[0])
                o1 = bpy.data.objects.get(self.picked_names[1])
                o2 = bpy.data.objects.get(self.picked_names[2])

                if not (o0 and o1 and o2):
                    self.report({'ERROR'}, "One or more picked objects no longer exist.")
                    return {'CANCELLED'}

                plane = create_plane_from_points(
                    o0.location.copy(),
                    o1.location.copy(),
                    o2.location.copy(),
                )

                if plane is None:
                    self.report({'ERROR'}, "Objects are collinear or too close; no plane created.")
                    return {'CANCELLED'}

                self.report({'INFO'}, f"Plane created: {plane.name}")
                return {'FINISHED'}

        # Let Blender handle other events normally (selection, navigation, etc.)
        return {'PASS_THROUGH'}


class VIEW3D_PT_plane_picker(bpy.types.Panel):
    """Panel in the 3D Viewport to start the plane picker"""
    bl_label = "Plane From 3 Objects"
    bl_idname = "VIEW3D_PT_plane_picker"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Plane Picker"

    def draw(self, context):
        layout = self.layout
        layout.operator(OBJECT_OT_pick_three_objects.bl_idname,
                        text="Pick 3 Objects for Plane")


# Optional: add to Object menu as well
def menu_func(self, context):
    self.layout.operator(OBJECT_OT_pick_three_objects.bl_idname,
                         text="Pick 3 Objects for Plane")


classes = (
    OBJECT_OT_pick_three_objects,
    VIEW3D_PT_plane_picker,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_object.append(menu_func)


def unregister():
    bpy.types.VIEW3D_MT_object.remove(menu_func)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()