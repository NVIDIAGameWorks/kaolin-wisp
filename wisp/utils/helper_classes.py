class DotDict(dict):
    """
    from https://stackoverflow.com/questions/13520421/recursive-dotdict

    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        if dct is not None:
            for key, value in dct.items():
                if hasattr(value, "keys"):
                    value = DotDict(value)
                self[key] = value
