Here is a list of examples that are added in the example folder:<br>
<b> A simple sphere</b><br>
<img src="readme_images/simple_sphere.png">
<br>

<code> sphere = Sphere(location=[0, 0, 0], radius=10,color="gold",smooth=3)</code><br>

<b> MultiSphere </b><br>
<img src="readme_images/multi_sphere.png">
<br>



Once you cloned the repository, you need to install the bpy library.<br>
This is what worked for me:<br>
<b>Linux</b><br>
<br>
activate your virtual environment (you should most conveniently use the python distributed with your blender installation)<br>
source venv/bin/activate.fish  (sorry, I use fish terminals)<br>
<br>
Then you can just run<br>
pip install python-dev-tools
pip install bpy<br>
pip install mathutils<br>
<br>
and the current bpy-wheels downloaded and bpy and mathutils are installed<br>
<br>
<b>Windows</b><br>
create a virtual environment (use the python distributed with you latest blender installation)<br>
Then you can just run<br>
pip install bpy<br>
pip install mathutils<br>
<br>
On Windows: You need to install the Visual C++ Build Tools and then install the python development package using pip<br>
pip install --upgrade setuptools<br>
pip install --upgrade wheel<br>
pip install --upgrade python-dev-tools (only works with pip 22.3)<br>
