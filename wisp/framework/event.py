# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from pydispatch import dispatcher

""" This module contains logic required to set up wisp's event handlers framework. """

# TODO(operel): Add "fire" event

##############################################
#            Public Handlers API             #
##############################################

def watch(watched_obj, field, status, handler):
    """ registers the handler for status updates on watched_obj.field.
        For example: watch(scene_status, "cam_controller", "changed", app.on_camera_controller_changed)
    """
    dispatcher.connect(handler, (status, field), sender=watched_obj)


def watchedfields(cls=None):
    """ Returns the class augmented with a custom __setattr__ implementation which notifies subscribers
        when class fields are updated.
    """
    def wrap(cls):
        return _register_func(cls)
    if cls is None:         # Called as @watchedfields()
        return wrap
    else:
        return wrap(cls)    # Called as @watchedfields

##############################################
#            watchedfields hooks             #
##############################################

def __setattr_notify__(obj, key, value):
    if hasattr(obj, key):
        prev_val = obj.__getattribute__(key) # First time this attribute is set
    else:
        prev_val = None
    obj._setattr(key, value)                 # Invoke internal setter here
    if prev_val != value:
        dispatcher.send(('changed', key), obj, value=value)

def _register_func(cls):
    # __setattr__ already explicitly defined, use it as internal setter implementation
    if '__setattr__' in cls.__dict__:
        setter_func = cls.__dict__['__setattr__']
    else:   # __setattr__ not defined, use the default implementation which simply sets the attribute
        def _setter_func(obj, key, value):
            obj.__dict__[key] = value
        setter_func = _setter_func
    setattr(cls, '_setattr', setter_func)
    setattr(cls, '__setattr__', __setattr_notify__)
    return cls


##############################################
#             watched iterables              #
##############################################

# TODO (operel): reconsider if we remove those, probably too slow but need to profile more carefully..

def wrap_iterable_fields(value):
    if isinstance(value, dict):
        return watcheddict(value)
    elif isinstance(value, list):
        return watchedlist(value)
    else:
        return value


class watcheddict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__ = type(dict.__name__, (self.__class__, dict), {})

    def __setitem__(self, item, value):
        super().__setitem__(item, value)
        dispatcher.send(('updated', self), self, value=item)


class watchedlist(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__ = type(list.__name__, (self.__class__, list), {})

    def __setitem__(self, item, value):
        super().__setitem__(item, value)
        dispatcher.send(('updated', self), self, value=item)
