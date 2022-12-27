# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Type, Any
from wisp.framework import WispState
from collections import deque
from wisp.core.colors import colors_generator, white, black, dark_gray, gray

# All wisp block widgets supported by the interactive renderer are registered here.
# Registration can be quickly achieved via the @widget decorator.
_WIDGETS_REGISTRY: Dict[Type[Any], Type[WidgetImgui]] = dict()

# A generator to yield the next unused color is a cyclical order, useful i.e. for colorful titles of blocks
_colors_gen = colors_generator(skip_colors=[white, black, dark_gray, gray])
next_unused_color = lambda: next(_colors_gen)


class WidgetImgui(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def paint(self, state: WispState, *args, **kwargs):
        raise NotImplementedError('imgui widgets must implement the paint method.')


def widget(wisp_block: Type[Any]):
    """ A decorator that registers a gui widget to paint the contents of a given wisp block.
        By registering a widget, the gui system knows how to load this widget when it traverses the
        scene graph & properties and encounters the wisp_block type.

        Users adding new wisp blocks can directly register corresponding widgets using this decorator.
    """
    def _register_widget_fn(widget_class: Type[WidgetImgui]):
        _WIDGETS_REGISTRY[wisp_block] = widget_class
        return widget_class
    return _register_widget_fn


def _lookup_widget(wisp_block: Any) -> Type[WidgetImgui]:
    """ Searches the wisp_block class hierarchy for any matches with some registered widget.
        wisp_block can be a
    """
    # Create a queue of base classes and search if any of them has a registered widget.
    # Basically this is a BFS on the base classes, start with the concrete class type..
    if not isinstance(wisp_block, type):    # Is this a type or an instance? Fetch the type
        wisp_block = type(wisp_block)
    base_queue = deque([wisp_block])

    widget_type = None
    while widget_type is None and base_queue:
        block_type = base_queue.popleft()
        widget_type = _WIDGETS_REGISTRY.get(block_type)
        if widget_type is None:
            bases = block_type.__bases__
            if len(bases) > 0:
                base_queue.extend(bases)
    return widget_type


def get_widget(wisp_block: Any) -> WidgetImgui:
    """ Return a widget which matches the given wisp block.
    A wisp block can be of any type / subtype which was registered with @widget.
    The lookup logic will first look for a widget registered under the type of wisp_block, and if it cannot find it,
    it will start looking up the hierarchy.
    Note that multiple-inheritance may result in undefined behavior in case of more than one match - in such cases
    users are encouraged to register some widget to the concrete type.
    Most wisp interfaces should already be registered to some general widget, so concrete types are not required
    to register a dedicated widget.
    However, they may register a specialized widget returned to display other forms of information.
    For example:

    Args:
        wisp_block: any object whose type / base types were registered with @widget.

    Returns:
        (WidgetImgui) imgui widget which matches the given wisp_block type
    """
    block_type = type(wisp_block)
    widget_type = _WIDGETS_REGISTRY.get(block_type)
    if widget_type is None:
        widget_type = _lookup_widget(block_type)
        _WIDGETS_REGISTRY[block_type] = widget_type

    if widget_type is None:
        raise ValueError(f'Gui cannot find a widget for {widget_type}. Make sure to register some widget with @widget, '
                         f'or consider if {widget_type} should subclass WispModule.')
    widget_instance = widget_type()  # Component widgets are assumed to take no args during construction
    return widget_instance
