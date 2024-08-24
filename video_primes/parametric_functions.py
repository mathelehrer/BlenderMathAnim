import math
from math import sin, cos, pi
import fnmatch
import bmesh
import bpy
from mathutils import Vector

context = bpy.context
scene = context.scene
link_object = scene.collection.objects.link if bpy.app.version >= (2, 80) else scene.objects.link
unlink_object = scene.collection.objects.unlink if bpy.app.version >= (2, 80) else scene.objects.unlink



#remove old labels and arrows
for ob in bpy.context.scene.objects:
    if fnmatch.fnmatch(ob.name,"XYZ Function*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Cylinder*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Plane*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"coordinate_system*"):
        ob.select_set(True)
    bpy.ops.object.delete()

def derive_bezier_handles(a, b, c, d, tb, tc):
    """
    Derives bezier handles by using the start and end of the curve with 2 intermediate
    points to use for interpolation.
    :param a:
        The start point.
    :param b:
        The first mid-point, located at `tb` on the bezier segment, where 0 < `tb` < 1.
    :param c:
        The second mid-point, located at `tc` on the bezier segment, where 0 < `tc` < 1.
    :param d:
        The end point.
    :param tb:
        The position of the first point in the bezier segment.
    :param tc:
        The position of the second point in the bezier segment.
    :return:
        A tuple of the two intermediate handles, that is, the right handle of the start point
        and the left handle of the end point.
    """

    # Calculate matrix coefficients
    matrix_a = 3 * math.pow(1 - tb, 2) * tb
    matrix_b = 3 * (1 - tb) * math.pow(tb, 2)
    matrix_c = 3 * math.pow(1 - tc, 2) * tc
    matrix_d = 3 * (1 - tc) * math.pow(tc, 2)

    # Calculate the matrix determinant
    matrix_determinant = 1 / ((matrix_a * matrix_d) - (matrix_b * matrix_c))

    # Calculate the components of the target position vector
    final_b = b - (math.pow(1 - tb, 3) * a) - (math.pow(tb, 3) * d)
    final_c = c - (math.pow(1 - tc, 3) * a) - (math.pow(tc, 3) * d)

    # Multiply the inversed matrix with the position vector to get the handle points
    bezier_b = matrix_determinant * ((matrix_d * final_b) + (-matrix_b * final_c))
    bezier_c = matrix_determinant * ((-matrix_c * final_b) + (matrix_a * final_c))

    # Return the handle points
    return (bezier_b, bezier_c)


def create_parametric_curve(
        function,
        *args,
        min: float = 0.0,
        max: float = 1.0,
        use_cubic: bool = True,
        iterations: int = 8,
        resolution_u: int = 10,
        **kwargs
    ):
    """
    Creates a Blender bezier curve object from a parametric function.
    This "plots" the function in 3D space from `min <= t <= max`.
    :param function:
        The function to plot as a Blender curve.
        This function should take in a float value of `t` and return a 3-item tuple or list
        of the X, Y and Z coordinates at that point:
        `function(t) -> (x, y, z)`
        `t` is plotted according to `min <= t <= max`, but if `use_cubic` is enabled, this function
        needs to be able to take values less than `min` and greater than `max`.
    :param *args:
        Additional positional arguments to be passed to the plotting function.
        These are not required.
    :param use_cubic:
        Whether or not to calculate the cubic bezier handles as to create smoother splines.
        Turning this off reduces calculation time and memory usage, but produces more jagged
        splines with sharp edges.
    :param iterations:
        The 'subdivisions' of the parametric to plot.
        Setting this higher produces more accurate curves but increases calculation time and
        memory usage.
    :param resolution_u:
        The preview surface resolution in the U direction of the bezier curve.
        Setting this to a higher value produces smoother curves in rendering, and increases the
        number of vertices the curve will get if converted into a mesh (e.g. for edge looping)
    :param **kwargs:
        Additional keyword arguments to be passed to the plotting function.
        These are not required.
    :return:
        The new Blender object.
    """

    # Create the Curve to populate with points.
    curve = bpy.data.curves.new('Parametric', type='CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 2

    # Add a new spline and give it the appropriate amount of points
    spline = curve.splines.new('BEZIER')
    spline.bezier_points.add(iterations)

    if use_cubic:
        points = [
            function(((i - 3) / (3 * iterations)) * (max - min) + min, *args, **kwargs)
            for i in range((3 * (iterations + 2)) + 1)
        ]

        # Convert intermediate points into handles
        for i in range(iterations + 2):
            a = points[(3 * i)]
            b = points[(3 * i) + 1]
            c = points[(3 * i) + 2]
            d = points[(3 * i) + 3]

            bezier_bx, bezier_cx = derive_bezier_handles(a[0], b[0], c[0], d[0], 1 / 3, 2 / 3)
            bezier_by, bezier_cy = derive_bezier_handles(a[1], b[1], c[1], d[1], 1 / 3, 2 / 3)
            bezier_bz, bezier_cz = derive_bezier_handles(a[2], b[2], c[2], d[2], 1 / 3, 2 / 3)

            points[(3 * i) + 1] = (bezier_bx, bezier_by, bezier_bz)
            points[(3 * i) + 2] = (bezier_cx, bezier_cy, bezier_cz)

        # Set point coordinates and handles
        for i in range(iterations + 1):
            spline.bezier_points[i].co = points[3 * (i + 1)]

            spline.bezier_points[i].handle_left_type = 'FREE'
            spline.bezier_points[i].handle_left = Vector(points[(3 * (i + 1)) - 1])

            spline.bezier_points[i].handle_right_type = 'FREE'
            spline.bezier_points[i].handle_right = Vector(points[(3 * (i + 1)) + 1])

    else:
        points = [function(i / iterations, *args, **kwargs) for i in range(iterations + 1)]

        # Set point coordinates, disable handles
        for i in range(iterations + 1):
            spline.bezier_points[i].co = points[i]
            spline.bezier_points[i].handle_left_type = 'VECTOR'
            spline.bezier_points[i].handle_right_type = 'VECTOR'

    # Create the Blender object and link it to the scene
    curve_object = bpy.data.objects.new('Parametric', curve)
    link_object(curve_object)

    # Return the new object
    return curve_object


def make_edge_loops(*objects):
    """
    Turns a set of Curve objects into meshes, creates vertex groups,
    and merges them into a set of edge loops.
    :param *objects:
        Positional arguments for each object to be converted and merged.
    """

    mesh_objects = []
    vertex_groups = []

    # Convert all curves to meshes
    for obj in objects:
        # Unlink old object
        unlink_object(obj)

        # Convert curve to a mesh
        if bpy.app.version >= (2, 80):
            new_mesh = obj.to_mesh().copy()
        else:
            new_mesh = obj.to_mesh(scene, False, 'PREVIEW')

        # Store name and matrix, then fully delete the old object
        name = obj.name
        matrix = obj.matrix_world
        bpy.data.objects.remove(obj)

        # Attach the new mesh to a new object with the old name
        new_object = bpy.data.objects.new(name, new_mesh)
        new_object.matrix_world = matrix

        # Make a new vertex group from all vertices on this mesh
        vertex_group = new_object.vertex_groups.new(name=name)
        vertex_group.add(range(len(new_mesh.vertices)), 1.0, 'ADD')

        vertex_groups.append(vertex_group)

        # Link our new object
        link_object(new_object)

        # Add it to our list
        mesh_objects.append(new_object)

    # Make a new context
    ctx = context.copy()

    # Select our objects in the context
    ctx['active_object'] = mesh_objects[0]
    ctx['selected_objects'] = mesh_objects
    if bpy.app.version >= (2, 80):
        ctx['selected_editable_objects'] = mesh_objects
    else:
        ctx['selected_editable_bases'] = [scene.object_bases[o.name] for o in mesh_objects]

    # Join them together
    bpy.ops.object.join(ctx)


def f(t, offset: float = 0.0):
    """
    The function to plot.
    :param t:
        The parametric variable, from `min <= t <= max`.
    :param offset:
        An extra offset parameter. These can be passed to create_parametric_curve.
    :return:
        A tuple of the (x, y, z) coordinate of the parametric at this position.
    """

    return (
        cos(4 * pi * t),
        sin(4 * pi * t),
        t + offset
    )


# Plot the parametric.

# A higher iteration count here adds more points, increasing accuracy but also
# increasing calculation time and memory usage.

# Setting use_cubic to false skips calculating handles, which saves time and memory
# but results in sharp edges in the spline.
curve = create_parametric_curve(f, offset=0.0, min=0.0, max=1.0, use_cubic=True, iterations=32)
#bpy.ops.object.convert(target='CURVE')