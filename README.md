<h1> Installation </h1>

Once you cloned the repository, you need to install the bpy library.<br>
This is what worked for me:<br>

<h3> Linux </h3>

activate your virtual environment (you should most conveniently use the python distributed with your blender installation)<br>
write something along that line:<br>
```source venv/bin/activate.fish```  
<br>
Then you need to run:<br>
```pip install bpy```<br>
```pip install mathutils```
<br>
because these libraries are not imported automatically from the requirement.txt
<br>
and the current bpy-wheelis downloaded and bpy and mathutils are installed<br>

