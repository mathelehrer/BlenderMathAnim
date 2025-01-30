import bpy

def level_tabs(n):
    out=''
    for i in range(n+1):
        out+='\t'
    return out

active = bpy.context.active_object
geo_tree = active.modifiers[0].node_group
print(geo_tree)
geo_nodes = geo_tree.nodes
print(geo_nodes)
geo_links = geo_tree.links
print(geo_links)

socket_dictionary = {}
node_dictionary = {}
node_count = 0
socket_count = 0

print("<NODE_XML>")
print(level_tabs(0),"<NODES>")
for node in geo_nodes:
    node_dictionary[node]=node_count
    print(level_tabs(1)+"<NODE id='"+str(node_count)+"' type='"+node.type+"' name='"+node.name+"' label='"+node.label+"' location='("+str(int(10*node.location[0]/200)/10)+","+str(int(10*node.location[1]/200)/10),end=")'")
    if hasattr(node,"hide"):
        print(" hide='"+str(node.hide),end="'")
    if hasattr(node,"mute"):
        print(" mute='"+str(node.mute),end="'")
    if hasattr(node,"transform_space"):
        print(" transform_space='"+str(node.transform_space),end="'")
    if hasattr(node,"domain"):
        print(" domain='"+str(node.domain),end="'")
        
    if hasattr(node,"component"):
        print(" component='"+str(node.component),end="'")
        
    if hasattr(node,"mode"):
        print(" mode='"+str(node.mode),end="'")
        
    if hasattr(node,"scale_mode"):
        print(" scale_mode='"+str(node.scale_mode),end="'")
    
    if hasattr(node,"data_type"):
        print(" data_type='"+str(node.data_type),end="'")
        
    if hasattr(node,"interpolation_type"):
        print(" interpolation_type='"+str(node.interpolation_type),end="'")
        
    if hasattr(node,"factor_mode"):
        print(" factor_mode='"+str(node.factor_mode),end="'")
        
    if hasattr(node,"operation"):
        print(" operation='"+str(node.operation),end="'")
        
    if hasattr(node,"use_clamp"):
        print(" use_clamp='"+str(node.use_clamp),end="'")
        
    if hasattr(node,"clamp_factor"):
        print(" clamp_factor'="+str(node.clamp_factor),end="'")
        
    if hasattr(node,"integer"):
        print(" integer='"+str(node.integer),end="'")
        
    if hasattr(node,"parent"):
        if node.parent is not None:
            print(" parent='"+str(node.parent.name),end="'")
        else:
            print(" parent='None'",end="")
        
    print(">")
    
    print(level_tabs(2)+"<INPUTS>")
    
    # inputs
    for input in node.inputs:
        socket_dictionary[input]=socket_count
        print(level_tabs(3),"<INPUT id='"+str(socket_count)+"' name='"+input.name+"' type='"+input.type+"' is_linked='"+str(input.is_linked),end="'")
        
        if not input.is_linked:
            if hasattr(input,"default_value"):
                print(" default_value='"+str(input.default_value),end="'")
        print("/>")
        socket_count+=1
    print(level_tabs(2),"</INPUTS>")
    
    print(level_tabs(2),"<OUTPUTS>")
    
    # outputs
    for output in node.outputs:
        socket_dictionary[output]=socket_count
        print(level_tabs(3),"<OUTPUT id='"+str(socket_count)+"' name='"+output.name+"' type='"+output.type+"' is_linked='"+str(output.is_linked),end="'")
        if hasattr(output,"default_value"):
            print(" default_value='"+str(output.default_value),end="'")
        
        print("/>")
        socket_count+=1
        
    print(level_tabs(2),"</OUTPUTS>")
    print(level_tabs(1),"</NODE>")
    node_count+=1

print(level_tabs(0),"</NODES>")
    
print(level_tabs(0),"<LINKS>")
for link in geo_links:
    print(level_tabs(1)+"<LINK from_node='"+str(node_dictionary[link.from_node])+"' from_socket='"+str(socket_dictionary[link.from_socket])+"' to_node='"+str(node_dictionary[link.to_node])+"' to_socket='"+str(socket_dictionary[link.to_socket])+"'/>")

print(level_tabs(0),"</LINKS>")
print("</NODE_XML>")