
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
