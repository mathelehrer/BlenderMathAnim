import bmesh
import numpy as np
import bpy
from mathutils import Vector, Quaternion

from objects.bobject import BObject
from utils.utils import quaternion_from_normal


class RevolutionSolid(BObject):
    """
    Create a solid of revolution with descent mesh:

    ex:
    paraboloid = RevolutionSolid(function=lambda x: np.sqrt(x),
                                     x_max=0.99, location=[0, 0, 0],
                                     resolution=64,
                                     color='fake_glass_drawing',
                                     scatter=0.5,shadow=False)

    The axis of revolution is the y-axis
    the function is provided as a lambda expression x->y=f(x)
    so far the domain is evaluated automatically from the provided value x_max
    """

    def __init__(self, function, x_max=1, resolution=10, **kwargs):
        self.kwargs = kwargs

        self.name= self.get_from_kwargs('name','RevolutionSolid')
        mesh =  self.create_mesh(function,x_max,resolution)

        super().__init__(name=self.name,mesh=mesh,**kwargs)

    def create_mesh(self,function=lambda x: x,x_max=1,resolution=10):
        vertices = []
        edges = []
        faces =[]

        first_row=True
        dx = x_max/resolution
        for i in range(0,resolution+1):
            x=i*dx
            y=function(x)

            #create vertices
            if i==0 and y!=0:
                vertices.append((x,0,0))
                first_row=False
                singular=True
            if y==0:
                vertices.append((x,0,0))
                singular=True
                singular_index =len(vertices)-1
            else:
                dphi = np.pi*2/resolution
                for j in range(0,resolution):
                    phi = j*dphi
                    vertices.append((x,y*np.cos(phi),y*np.sin(phi)))
                    v = len(vertices)
                    if j>0:
                        edges.append([v-1,v-2])
                    if not first_row:
                        if not singular:
                            edges.append([v-1,v-resolution-1])
                        else:
                            edges.append([v-1,singular_index])
                    if i>0 and j>0:
                        if not singular:
                            faces.append([v-1,v-2,v-2-resolution,v-1-resolution])
                        else:
                            faces.append([v-1,v-2,singular_index])

                edges.append([v-1,v-resolution])
                if not singular:
                    faces.append([v-resolution,v-1,v-1-resolution,v-2*resolution])
                else:
                    faces.append([v-resolution,v-1,singular_index])
                singular = False
            if i==resolution and y!=0:
                # final disk
                vertices.append([x,0,0])
                v=len(vertices)-1
                for j in range(0,resolution):
                    edges.append([v,v-j-1])
                    faces.append([v,v-(j+1)%resolution-1,v-j-1])

        new_mesh = bpy.data.meshes.new(self.name+'_mesh')
        new_mesh.from_pydata(vertices,edges,faces)
        new_mesh.update()
        return new_mesh









