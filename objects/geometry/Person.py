from objects.geometry.geo_bobject import GeoBObject


class Person(GeoBObject):
    def __init__(self,**kwargs):
        bob = GeoBObject.from_file("Person",with_wrapper=False,**kwargs)
        super().__init__(obj=bob,**kwargs)