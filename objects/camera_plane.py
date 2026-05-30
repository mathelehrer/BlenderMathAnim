from interface import ibpy
from interface.ibpy import to_vector
from objects.empties import EmptyAxes


class CameraPlane(EmptyAxes):
    """A plain-axes empty intended to act as a camera-aligned parent.

    Children parented to a :class:`CameraPlane` follow the camera; use
    :meth:`align_with_camera_location` to update the plane's rotation
    whenever the camera moves.
    """

    def __init__(self,**kwargs):
        """Create a camera-aligned empty.

        Args:
            **kwargs: Forwarded to :class:`EmptyAxes`. Object name is
                fixed to ``'CameraPlane'``.
        """
        super().__init__(name="CameraPlane",**kwargs)

    def add(self,bob):
        ibpy.set_parent(bob,self)

    def align_with_camera_location(self, location):
        """
        align the empty to the location of the camera
        """

        # get own location
        own_location = self.ref_obj.location
        # calculate direction to camera
        direction = to_vector(location) - own_location
        direction.normalize()

        # no rotation would be needed, when the camera direction was (0,0,1)
        self.ref_obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()

