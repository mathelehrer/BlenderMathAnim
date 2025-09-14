from interface.ibpy import create_mesh, get_geometry_node_from_modifier, change_default_value, change_default_integer, \
    get_node_from_shader
from mathematics.geometry.cell600 import get_3D_geometry, get_base_cell
from objects.bobject import BObject
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE
from utils.kwargs import get_from_kwargs


class Poly600Cell(BObject):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', '600Cell')

        # the data for this polytope is generated from two quaternions
        # the computations are performed in the file "mathematics/geometry/cell600.py"

        offset=get_from_kwargs(kwargs,"offset",0.1)
        vertices,edges,faces = get_3D_geometry(offset=offset)

        super().__init__(mesh=create_mesh(vertices, edges,faces), name=self.name, **kwargs)

    def grow(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,pause = DEFAULT_ANIMATION_TIME,modifier=None, **kwargs):
        if modifier is None:
            return super().appear(begin_time=begin_time, transition_time=transition_time, **kwargs)
        else:
            super().appear(begin_time=begin_time, transition_time=0)
            w_min_node=get_geometry_node_from_modifier(modifier,"WMin")
            w_max_node=get_geometry_node_from_modifier(modifier,"WMax")
            idx_range_node=get_geometry_node_from_modifier(modifier,"IndexRange")


            range_steps = [-1,-0.3,0.01,0.31,0.51,0.81,1.1,2]
            dt = transition_time/len(range_steps)
            dp = pause/len(range_steps)

            t0 = begin_time
            change_default_value(w_min_node,from_value=-1,to_value=range_steps[0],begin_time=begin_time,transition_time=0)
            change_default_value(w_max_node,from_value=-1,to_value=range_steps[1],begin_time=begin_time,transition_time=0)

            for i in range(1,len(range_steps)-1):
                t0 = change_default_integer(idx_range_node, from_value=0, to_value=340, begin_time=t0,
                                       transition_time=dt)

                change_default_value(w_min_node,from_value=range_steps[i-1],to_value=range_steps[i],begin_time=t0,transition_time=0)
                change_default_value(w_max_node,from_value=range_steps[i],to_value=range_steps[i+1],begin_time=t0,transition_time=0)
                change_default_integer(idx_range_node,from_value=340,to_value=0,begin_time=t0,transition_time=0)
                t0 += dp

        return begin_time+transition_time+pause


    def rotate(self,angle,from_angle,to_angle,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,modifier=None):
        if modifier:
            angle_node = get_geometry_node_from_modifier(modifier,angle)
            change_default_value(angle_node,from_value=from_angle,to_value=to_angle,begin_time=begin_time,transition_time=transition_time)
            return begin_time+transition_time

    def change_alpha(self,from_value=1,to_value=0,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,modifier=None):
        if modifier:
            face_material = modifier.materials[0]
            alpha_node = get_node_from_shader(face_material, "AlphaMultiplier")
            change_default_value(alpha_node, from_value=from_value, to_value=to_value, begin_time=begin_time,
                                      transition_time=transition_time)
            return begin_time+transition_time