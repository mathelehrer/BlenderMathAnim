from objects.plane import Plane


class Floor(Plane):
    def __init__(self, u=[-1, 1], v=[-1, 1], location=None, resolution=10,color='checker', **kwargs):
        super().__init__(u=u,v=v,location=location,resolution=resolution,name='floor',color=color,**kwargs)
