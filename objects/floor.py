from objects.plane import Plane


class Floor(Plane):
    """A floor :class:`Plane` with a checkerboard shader by default."""

    def __init__(self, u=[-1, 1], v=[-1, 1], location=None, resolution=10,color='checker', **kwargs):
        """Create a floor plane.

        Args:
            u: ``[u_min, u_max]`` extent along X. Defaults to ``[-1, 1]``.
            v: ``[v_min, v_max]`` extent along Y. Defaults to ``[-1, 1]``.
            location: World location. Defaults to the plane's auto-centre.
            resolution: Subdivisions per axis. Defaults to 10.
            color: Material name. Defaults to ``'checker'``.
            **kwargs: Forwarded to :class:`Plane`. Object name is fixed
                to ``'floor'``.
        """
        super().__init__(u=u,v=v,location=location,resolution=resolution,name='floor',color=color,**kwargs)
