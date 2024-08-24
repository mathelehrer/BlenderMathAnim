import bpy

bpy.ops.mesh.primitive_xyz_function_surface(x_eq="cos(u)*((v>-2)*(v<0)*0.5+(v>=0)*(1-v))",
 z_eq="v",y_eq="sin(u)*((v>-2)*(v<0)*0.5+(v>=0)*(1-v))", range_v_min=-2, range_v_max=1, range_v_step=32)
arrow = bpy.context.active_object
arrow.location[2]=3
arrow.rotation_euler[0]=3.14159
