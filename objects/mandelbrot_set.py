from objects.plane import Plane

class MandelbrotSet(Plane):
    """A :class:`Plane` pre-configured with the ``'mandelbrot'`` shader
    material, sized to cover the standard Mandelbrot viewing window."""

    def __init__(self,u=[-2.1,0.5],v=[-1.3,1.3],**kwargs):
        """Create a Mandelbrot-rendering plane.

        Args:
            u: ``[u_min, u_max]`` extent along the real axis. Defaults
                to ``[-2.1, 0.5]`` (classic Mandelbrot view).
            v: ``[v_min, v_max]`` extent along the imaginary axis.
                Defaults to ``[-1.3, 1.3]``.
            **kwargs: Forwarded to :class:`Plane`. ``color`` is forced
                to ``'mandelbrot'`` -- the underlying shader iterates
                ``z -> z**2 + c`` per pixel.
        """
        super().__init__(u=u,v=v,color='mandelbrot',**kwargs)