import bpy

from bpy import context, data, ops
import numpy as np
import bmesh
import mathutils
import fnmatch
from pathlib import Path
from mathutils import Vector

import os
import glob
import subprocess
import tempfile
import shutil
import math

from math import sin, cos, pi
import fnmatch


bpy.ops.curve.primitive_nurbs_path_add(radius=1, enter_editmode=False, align='WORLD', location=(1, 0, 1), rotation=(0, 0, 1.5708))
path = bpy.context.active_object
path.scale[0]=20


bpy.ops.mesh.primitive_xyz_function_surface(x_eq="cos(u)*((v>-2)*(v<0)*0.25+1*(v>=0)*(1-v))",
z_eq="v",
y_eq="sin(u)*((v>-2)*(v<0)*0.25+1*(v>=0)*(1-v))", range_v_min=-2, range_v_max=1, range_v_step=32)
bpy.ops.object.editmode_toggle()

#put pivot point at the end of the arrow
xarrow = bpy.context.active_object
bpy.context.scene.cursor.location = (0.0, 0.0, -2.0)
bpy.ops.object.origin_set(type='ORIGIN_CURSOR',center ='MEDIAN')
bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
bpy.ops.object.constraint_add(type='FOLLOW_PATH')
constraint = bpy.context.object.constraints["Follow Path"]
constraint.target = path
constraint.use_curve_follow=True
bpy.ops.constraint.followpath_path_animate(constraint="Follow Path", owner='OBJECT')


arrow_scale = 1



xarrow.rotation_euler[1]=3.14159/2
#xarrow.scale[0]=arrow_scale
#xarrow.scale[1]=arrow_scale
xarrow.scale[2]=2*arrow_scale
xarrow.location[1]=0
xarrow.location[0]=1
xarrow.location[2]=1

