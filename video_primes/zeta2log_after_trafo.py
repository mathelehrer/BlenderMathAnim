import bpy

from pathlib import Path



args = []
path = "/home/jmartin/Dropbox/MyBlender/PrimesAndZeta/log_zeta_args_-50..50_10000.csv"  # a blender relative path
f = Path(bpy.path.abspath(path)) 

if f.exists():
    text  = f.read_text()
    lines = text.split('\n')
    for line in lines:
            if len(line)>0:
                data = float(line)
                args.append(data)


def arg_log_zeta(pos):
    index = int((pos+50)/100*10000)
    return args[index]

bpy.app.driver_namespace['arg_log_zeta']=arg_log_zeta




camera = bpy.data.objects["Camera"]
camera.data.lens =30


camera.constraints["Follow Path"].use_fixed_location = False
camera.constraints["Follow Path"].use_curve_follow =False
camera.constraints["Follow Path"].target=None

bpy.ops.mesh.primitive_xyz_function_surface(y_eq="cos(u)*((v>-2)*(v<0)*0.15+0.3*(v>=0)*(1-v))",
x_eq="v",
z_eq="sin(u)*((v>-2)*(v<0)*0.15+0.3*(v>=0)*(1-v))", range_v_min=-2, range_v_max=1, range_v_step=32)
bpy.ops.object.editmode_toggle()

#put pivot point at the end of the arrow
xarrow = bpy.context.active_object
bpy.context.scene.cursor.location = (-2.0, 0.0, 0.0)
bpy.ops.object.origin_set(type='ORIGIN_CURSOR',center ='MEDIAN')
bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)


bpy.context.scene.frame_end = 3000
frame = 1501
bpy.context.scene.frame_set(frame)
camera.location = (8,-60,5)
camera.keyframe_insert("location")

frame = frame +30
bpy.context.scene.frame_set(frame)
camera.location =(8,-60,15)
camera.keyframe_insert("location")

