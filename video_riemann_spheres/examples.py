import bpy
import numpy as np

bpy.ops.mesh.riemann_sphere_add(sub_num=128, radius=3, function="(z*z-9)/(z-1)")
blob = bpy.context.active_object
blob.location[1]=10

bpy.ops.mesh.riemann_sphere_add(sub_num=128, radius=3, function="mandelbrot(z)")
blob = bpy.context.active_object
blob.location[1]=10
