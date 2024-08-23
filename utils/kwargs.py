def get_from_kwargs(kwargs,attr,default_value):
    if attr in kwargs:
        return kwargs.pop(attr)
    else:
        return default_value

