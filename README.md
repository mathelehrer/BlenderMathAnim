This library is the backbone of the video projects that you can find on my youtube channel: https://www.youtube.com/@Number_Cruncher<br>
All animations are rendered using the Blender render engine. However, the blender files are not created in blender, but rather in python using the bpy library. This makes it easy to combine the workflow with version control systems like git. The library is work in progress and I am constantly adding new features and improving the existing ones.
The following gallery shows objects that are created from simple spheres. You can find more examples in the examples folder. I try to include more and more examples to make it easier for you to learn the usage of the library.

<br>
<h1> Examples </h1>
Here is a gallery of examples that are you can find in the example folder:<br>
<b> A simple sphere</b><br>
<img src="readme_images/simple_sphere.png">
<br>

<code> sphere = Sphere(location=[0, 0, 0], radius=10,color="gold",smooth=3)</code><br>

<b> MultiSphere </b><br>
<img src="readme_images/multi_sphere.png">
<br>

<b> HalfSphere </b><br>
<img src="readme_images/half_sphere.png">
<br>
<code> 
half_sphere = HalfSphere(1 / 2, location=[0, 0, 1 / 2], resolution=100, solid=0.01, offset=0,color="example",roughness=0,metallic=0.5)
</code><br>

<b> StackOfSpheres </b><br>
<img src="readme_images/stack_of_spheres.png">
<br>
<code>
sequence = [50, 20, 20, 20, 3, 1, 1]
stack_b = StackOfSpheres(radius=0.5, number_of_spheres=100, color='important', smooth=2,
                             location=[2.4894, 2.8364, 0.116], name="StackBob", scale=2)
</code><br>


<h1> Installation </h1>

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
<br>
and the current bpy-wheels downloaded and bpy and mathutils are installed<br>
<br>
<b>Windows</b><br>
create a virtual environment (use the python distributed with you latest blender installation)<br>
Then you can just run<br>
pip install bpy<br>
<br>
On Windows: You need to install the Visual C++ Build Tools and then install the python development package using pip<br>
pip install --upgrade setuptools<br>
pip install --upgrade wheel<br>
pip install --upgrade python-dev-tools (only works with pip 22.3)<br>
