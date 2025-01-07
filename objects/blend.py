from objects.bobject import BObject

class Blend(BObject):
    '''
    Create a blend that can be used in the camera plane

    '''
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        name=self.get_from_kwargs('name','Blend')
        blend = BObject.from_file("Blend",name=name)

        super().__init__(obj=blend.ref_obj,name=name,**kwargs)

