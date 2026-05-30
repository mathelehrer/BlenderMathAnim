from objects.geometry.geo_bobject import GeoBObject


class Person(GeoBObject):
    """A human-figure model loaded from the bundled ``Person`` asset."""

    def __init__(self,**kwargs):
        """Load the bundled person model.

        Args:
            **kwargs: Forwarded to :class:`GeoBObject`. Standard
                appearance/transform kwargs.
        """
        bob = GeoBObject.from_file("Person",with_wrapper=False,**kwargs)
        super().__init__(obj=bob,**kwargs)