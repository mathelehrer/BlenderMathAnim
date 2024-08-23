from objects.plane import Plane

class MandelbrotSet(Plane):
    def __init__(self,u=[-2.1,0.5],v=[-1.3,1.3],**kwargs):
        super().__init__(u=u,v=v,color='mandelbrot',**kwargs)