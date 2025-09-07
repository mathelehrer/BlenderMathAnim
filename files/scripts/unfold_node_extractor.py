import bpy
from mathutils import Euler,Vector

def level_tabs(n):
    out=""
    for i in range(n+1):
        out+="\t"
    return out

active = bpy.context.active_object
geo_tree = active.modifiers[0].node_group
geo_nodes = geo_tree.nodes
geo_links = geo_tree.links

socket_dictionary = {}
node_dictionary = {}
node_count = 0
socket_count = 0


def get_default_value(value):
    if isinstance(value,bpy.types.bpy_prop_array):
        return str(list(value[0:len(value)]))
    elif isinstance(value,Euler):
        return str(list(value[0:3]))
    elif isinstance(value,Vector):
        return str(list(value[0:len(value)]))
    return str(value)    


def append(word,end="\n"):
    data = ""
    data+=word
    data+=end
    return data
    

filedata = ""
    
filedata+=append('<NODE_XML>')
filedata+=append(level_tabs(0)+'<NODES>')
for node in geo_nodes:
    node_dictionary[node]=node_count
    filedata+=append(level_tabs(1)+'<NODE id="'+str(node_count)+'" type="'+node.type+'" name="'+node.name+'" label="'+node.label+
    '" location="('+str(int(10*node.location[0]/200)/10)+','+str(int(10*node.location[1]/200)/10),end=')"')
    
    if hasattr(node,'axis'):
        filedata+=append(' axis="'+str(node.axis),end='"')
        
    if hasattr(node,'primary_axis'):
        filedata+=append(' primary_axis="'+str(node.primary_axis),end='"')
        
    if hasattr(node,'secondary_axis'):
        filedata+=append(' secondary_axis="'+str(node.secondary_axis),end='"')  
    
    if hasattr(node,'pivot_axis'):
        filedata+=append(' pivot_axis="'+str(node.pivot_axis),end='"')
    
    if hasattr(node,'hide'):
        filedata+=append(' hide="'+str(node.hide),end='"')
    
    if hasattr(node,'mute'):
        filedata+=append(' mute="'+str(node.mute),end='"')
    
    if hasattr(node,'transform_space'):
        filedata+=append(' transform_space="'+str(node.transform_space),end='"')
    
    if hasattr(node,'rotation_space'):
        filedata+=append(' rotation_space="'+str(node.rotation_space),end='"')
    
    if hasattr(node,'domain'):
        filedata+=append(' domain="'+str(node.domain),end='"')
        
    if hasattr(node,'component'):
        filedata+=append(' component="'+str(node.component),end='"')
        
    if hasattr(node,'mode'):
        filedata+=append(' mode="'+str(node.mode),end='"')
        
    if hasattr(node,'scale_mode'):
        filedata+=append(' scale_mode="'+str(node.scale_mode),end='"')
    
    if hasattr(node,'data_type'):
        filedata+=append(' data_type="'+str(node.data_type),end='"')
        
    if hasattr(node,'input_type'):
        filedata+=append(' input_type="'+str(node.input_type),end='"')
        
    if hasattr(node,'interpolation_type'):
        filedata+=append(' interpolation_type="'+str(node.interpolation_type),end='"')
        
    if hasattr(node,'factor_mode'):
        filedata+=append(' factor_mode="'+str(node.factor_mode),end='"')
        
    if hasattr(node,'operation'):
        filedata+=append(' operation="'+str(node.operation),end='"')
        
    if hasattr(node,'use_clamp'):
        filedata+=append(' use_clamp="'+str(node.use_clamp),end='"')
        
    if hasattr(node,'clamp_factor'):
        filedata+=append(' clamp_factor="'+str(node.clamp_factor),end='"')
        
    if hasattr(node,'integer'):
        filedata+=append(' integer="'+str(node.integer),end='"')
        
    if hasattr(node,'parent'):
        if node.parent is not None:
            filedata+=append(' parent="'+str(node.parent.name),end='"')
        else:
            filedata+=append(' parent="None"',end='')
        
    filedata+=append('>')
    
    filedata+=append(level_tabs(2)+'<INPUTS>')
    
    # inputs
    for input in node.inputs:
        socket_dictionary[input]=socket_count
        filedata+=append(level_tabs(3)+'<INPUT id="'+str(socket_count)+'" name="'+input.name+'" type="'+input.type+'" is_linked="'+str(input.is_linked),end='"')
        
        if not input.is_linked:
            if hasattr(input,'default_value'):
                filedata+=append(' default_value="' + get_default_value(input.default_value),end='"')
        filedata+=append('/>')
        socket_count+=1
    filedata+=append(level_tabs(2)+'</INPUTS>')
    
    filedata+=append(level_tabs(2)+'<OUTPUTS>')
    
    # outputs
    for output in node.outputs:
        socket_dictionary[output]=socket_count
        filedata+=append(level_tabs(3)+'<OUTPUT id="'+str(socket_count)+'" name="'+output.name+'" type="'+output.type+'" is_linked="'+str(output.is_linked),end='"')
        if hasattr(output,'default_value'):
            filedata+=append(' default_value="' + get_default_value(output.default_value),end='"')
        
        filedata+=append('/>')
        socket_count+=1
        
    filedata+=append(level_tabs(2)+'</OUTPUTS>')
    filedata+=append(level_tabs(1)+'</NODE>')
    node_count+=1

filedata+=append(level_tabs(0)+'</NODES>')
    
filedata+=append(level_tabs(0)+'<LINKS>')
for link in geo_links:
    filedata+=append(level_tabs(1)+'<LINK from_node="'+str(node_dictionary[link.from_node])+'" from_socket="'+str(socket_dictionary[link.from_socket])+'" to_node="'+str(node_dictionary[link.to_node])+'" to_socket="'+str(socket_dictionary[link.to_socket])+'"/>')

filedata+=append(level_tabs(0)+'</LINKS>')
filedata+=append('</NODE_XML>')


print(filedata)

# store data to file

with open("/files2/xml/unfolding_node.xml", 'w') as f:
    print("success")
    f.write(filedata)