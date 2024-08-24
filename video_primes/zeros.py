import bpy
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


def get_coordinate_material(color = (0.013,0.8,0,1)):
    mat = bpy.data.materials.new("material")

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    output = nodes.get("Material Output")
    output.location = Vector((600,200))
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs[18].default_value=0.1  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs[17].default_value =color
    bsdf.inputs[0].default_value=color
    links =mat.node_tree.links
    links.new(bsdf.outputs[0],output.inputs[0])
    return mat




# Imports compiled latex code into blender given chosen settings.
def import_latex(context, latex_code, text_scale, x_loc, y_loc, z_loc, x_rot, y_rot, z_rot, custom_preamble_bool, temp_dir, filename='temp',preamble_path=None):

    # Set current directory to temp_directory
    current_dir = os.getcwd()
    os.chdir(temp_dir)

    # Create temp latex file with specified preamble.
    temp_file_name = temp_dir + os.sep + filename
    if custom_preamble_bool:
        shutil.copy(preamble_path, temp_file_name + '.tex')
        temp = open(temp_file_name + '.tex', "a")
    else:
        temp = open(temp_file_name + '.tex', "a")
        default_preamble = '\\documentclass{amsart}\n\\usepackage{amssymb,amsfonts}\n\\usepackage{tikz}' \
                           '\n\\usepackage{tikz-cd}\n\\pagestyle{empty}\n\\thispagestyle{empty}'
        temp.write(default_preamble)

    # Add latex code to temp.tex and close the file.
    temp.write('\n\\begin{document}\n' + latex_code + '\n\\end{document}')
    temp.close()

    # Try to compile temp.tex and create an svg file
    try:
        # Updates 'PATH' to include reference to folder containing latex and dvisvgm executable files.
        # This only matters when running on MacOS. It is unnecessary for Linux and Windows.
        latex_exec_path = '/Library/TeX/texbin'
        local_env = os.environ.copy()
        local_env['PATH'] = (latex_exec_path + os.pathsep + local_env['PATH'])

        subprocess.call(["latex", "-interaction=batchmode", temp_file_name + ".tex"], env=local_env)
        subprocess.call(["dvisvgm", "--no-fonts", temp_file_name + ".dvi"], env=local_env)

        objects_before_import = bpy.data.objects[:]

        bpy.ops.object.select_all(action='DESELECT')


        svg_file_list = glob.glob("*.svg")

        if len(svg_file_list) == 0:
            ErrorMessageBox("Please check your latex code for errors and that latex and dvisvgm are properly "
                            "installed. Also, if using a custom preamble, check that it is formatted correctly.",
                            "Compilation Error")
        else:
            # Import svg into blender as curve
            file =""
            for svg in svg_file_list:
                if filename in svg:
                    file=svg
                    print(file," ",svg)
            svg_file_path = temp_dir + os.sep + file
            bpy.ops.import_curve.svg(filepath=svg_file_path)

            # Select imported objects
            imported_curve = [x for x in bpy.data.objects if x not in objects_before_import]
            active_obj = imported_curve[0]
            context.view_layer.objects.active = active_obj
            for x in imported_curve:
                x.select_set(True)

            # Convert to Mesh
            bpy.ops.object.convert(target='MESH')

            # Adjust scale, location, and rotation.
            bpy.ops.object.join()
            bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
            active_obj.scale = (600*text_scale, 600*text_scale, 600*text_scale)
            active_obj.location = (x_loc, y_loc, z_loc)
            active_obj.rotation_euler = (math.radians(x_rot), math.radians(y_rot), math.radians(z_rot))
            # Move mesh to scene collection and delete the temp.svg collection. Then rename mesh.
            temp_svg_collection = active_obj.users_collection[0]
            bpy.ops.object.move_to_collection(collection_index=0)
            bpy.data.collections.remove(temp_svg_collection)
            active_obj.name = 'Latex Figure'
    except FileNotFoundError as e:
        ErrorMessageBox("Please check that LaTeX is installed on your system.", "Compilation Error")
    except subprocess.CalledProcessError:
        ErrorMessageBox("Please check your latex code for errors and that latex and dvisvgm are properly installed. "
                        "Also, if using a custom preamble, check that it is formatted correctly.", "Compilation Error")
    finally:
        os.chdir(current_dir)
        print("Finished trying to compile latex and create an svg file.")



#remove old labels and arrows
for ob in bpy.context.scene.objects:
    if fnmatch.fnmatch(ob.name, "Latex*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"XYZ Function*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Cylinder*"):
        ob.select_set(True)
    bpy.ops.object.delete()

#path = some_object.filepath # as declared by props
zeros =[]
path = "/home/jmartin/Dropbox/MyBlender/PrimesAndZeta/zeros.dat"  # a blender relative path
f = Path(bpy.path.abspath(path)) # make a path object of abs path

if f.exists():
    text  = f.read_text()
    lines = text.split('\n')
    for line in lines:
        zeros.append(float(line))
else:
    print("No file found")
 
#line onehalf    
bpy.ops.mesh.primitive_cylinder_add(radius=0.075,depth=100)
cylinder = bpy.context.active_object
cylinder.location[0]=-5
cylinder.rotation_euler[0]=3.14159/2
cylinder.scale[2]=1.5
cylinder.data.materials.append(get_coordinate_material())


arrow_z0 = 43
text_z0 = 40.05

for (i,zero) in enumerate(zeros):
    if i<10:
        print(i," ",zero)
        zero_str = str(int(zero*100+0.5)/100)
        bpy.context.scene.my_tool.latex_code="\\tfrac{1}{2}+"+zero_str+"i"
        bpy.context.scene.my_tool.text_scale=2
        bpy.context.scene.my_tool.x_rot=90
        bpy.context.scene.my_tool.y_loc = zero
        bpy.context.scene.my_tool.x_loc = 0
        bpy.context.scene.my_tool.z_loc = text_z0
        scene = bpy.context.scene
        t= scene.my_tool
        import_latex(bpy.context,t.latex_code,t.text_scale,t.x_loc,t.y_loc,t.z_loc,t.x_rot,t.y_rot,t.z_rot,False,'/home/jmartin/working_dir/','zero+'+str(i))
        
        bpy.ops.mesh.primitive_xyz_function_surface(x_eq="cos(u)*((v>-2)*(v<0)*0.25+0.5*(v>=0)*(1-v))",z_eq="v",y_eq="sin(u)*((v>-2)*(v<0)*0.25+0.5*(v>=0)*(1-v))", range_v_min=-2, range_v_max=1, range_v_step=32)
        bpy.ops.object.editmode_toggle()
        arrow = bpy.context.active_object
        arrow.location[2]=arrow_z0
        arrow.location[1]=zero
        arrow.location[0]=-4.75
        arrow.rotation_euler[0]=3.14159
        
        bpy.context.scene.my_tool.latex_code="\\tfrac{1}{2}-"+zero_str+"i"
        bpy.context.scene.my_tool.text_scale=2
        bpy.context.scene.my_tool.x_rot=90
        bpy.context.scene.my_tool.z_rot=0
        bpy.context.scene.my_tool.x_loc = 0
        bpy.context.scene.my_tool.y_loc = -zero
        bpy.context.scene.my_tool.z_loc = text_z0
        scene = bpy.context.scene
        t= scene.my_tool
        import_latex(bpy.context,t.latex_code,t.text_scale,t.x_loc,t.y_loc,t.z_loc,t.x_rot,t.y_rot,t.z_rot,False,'/home/jmartin/working_dir/','zero-'+str(i))
        
        bpy.ops.mesh.primitive_xyz_function_surface(x_eq="cos(u)*((v>-2)*(v<0)*0.25+0.5*(v>=0)*(1-v))",z_eq="v",y_eq="sin(u)*((v>-2)*(v<0)*0.25+(v>=0)*0.5*(1-v))", range_v_min=-2, range_v_max=1, range_v_step=32)
        bpy.ops.object.editmode_toggle()
        arrow = bpy.context.active_object
        arrow.location[2]=arrow_z0
        arrow.location[1]=-zero
        arrow.location[0]=-4.75
        arrow.rotation_euler[0]=3.14159
        arrow.data.materials.append(get_coordinate_material())


arrows = []
labels = []
for ob in bpy.context.scene.objects:
    if fnmatch.fnmatch(ob.name, "Latex*"):
        labels.append(ob)
        m = ob.modifiers.new("Solidify", type='SOLIDIFY')
        m.thickness = 0.0002
        m.offset=0
        m.use_even_offset = True
        m.use_quality_normals = True
        m.use_rim = True
        bpy.context.view_layer.objects.active = ob
        #bpy.ops.object.modifier_apply(modifier="Solidify")
        ob.data.materials.clear()
        ob.data.materials.append(get_coordinate_material(color = (0,0.25,0,1)))
    if fnmatch.fnmatch(ob.name,"XYZ Function*"):
        arrows.append(ob)
        ob.data.materials.append(get_coordinate_material())
        

# set first and last frame index
total_time = 33 # Animation should be 2*pi seconds long
fps =30  # Frames per second (fps)
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = int(total_time*fps)+1

offset = 280

bpy.context.scene.frame_set(offset+1)
cylinder.scale[2]=0
cylinder.keyframe_insert("scale")

bpy.context.scene.frame_set(offset+31)
cylinder.scale[2]=1.5
cylinder.keyframe_insert("scale")

offset = 310
i = 0
for arrow,label in zip(arrows,labels):
    bpy.context.scene.frame_set(offset+20*i+31)
    arrow.keyframe_insert("location")
    label.keyframe_insert("location")
    bpy.context.scene.frame_set(offset+20*i+61-5)
    label.keyframe_insert("rotation_euler")
    bpy.context.scene.frame_set(offset+20*i+61)
    arrow.location.z=arrow.location.z-40
    label.location.z=label.location.z-40
    label.rotation_euler[0]=0
    arrow.keyframe_insert("location")
    label.keyframe_insert("location")
    label.keyframe_insert("rotation_euler")
    i=i+0.5
    
    
offset = 570

for arrow,label in zip(arrows,labels):
    bpy.context.scene.frame_set(offset)
    arrow.keyframe_insert("location")
    bpy.context.scene.frame_set(offset+30)
    arrow.location.z=arrow.location.z+10
    arrow.keyframe_insert("location")
    
#line of integration
bpy.ops.mesh.primitive_cylinder_add(radius=0.15,depth=100)
cylinder = bpy.context.active_object
cylinder.rotation_euler[0]=3.14159/2
cylinder.location[2]=1.4
cylinder.scale[2]=0
cylinder.data.materials.append(get_coordinate_material(color = (0.35,0.35,0.35,1)))

offset = 630
bpy.context.scene.frame_set(offset)
cylinder.scale[2]=0
cylinder.keyframe_insert("scale")

for i,zero in enumerate(zeros):
    if i<10:
        bpy.context.scene.frame_set(offset+30*(i+1)-3)
        cylinder.keyframe_insert("scale")
        bpy.context.scene.frame_set(offset+30*(i+1))
        cylinder.scale[2]=2*zero/100
        cylinder.keyframe_insert("scale")
    
