from objects.geometry.sphere import Sphere


class Point(Sphere):
    """
    Create a point, whose location on the screen is associated with a coordinate system
    """

    def __init__(self, coordinate_system, coordinates=(0,0,0), size = 1, material='example', **kwargs):
        self.kwargs = kwargs
        name = self.get_from_kwargs('name','P')
        self.kwargs.pop('name')
        self.coordinate_system=coordinate_system
        location = self.coordinate_system.coords2location(coordinates)
        super().__init__(size/10,location=location, name=name,material=material, **kwargs)
        self.coordinate_system.add_object(self)

