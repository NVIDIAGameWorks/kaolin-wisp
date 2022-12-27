# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from wisp.framework import WispState
from wisp.renderer.core.api import RayTracedRenderer
from .widget_imgui import WidgetImgui, widget, get_widget
from .widget_property_editor import WidgetPropertyEditor


@widget(RayTracedRenderer)
class WidgetRayTracedRenderer(WidgetImgui):
    """ A default widget for ray tracing based bottom-level-renderers.
    If you added a new type of neural field and specialized a new type of RayTracedRenderer / BottomLevelRenderer,
    but not a widget, this default widget will be loaded as a fallback.
    The neural field information will be displayed, but no assumptions on the inference-time tracer can be made,
    therefore tracer fields will not be displayed.
    """

    def __init__(self):
        super().__init__()
        self.properties_widget = WidgetPropertyEditor()

    def paint(self, state: WispState, renderer: RayTracedRenderer = None, *args, **kwargs):
        if renderer is None:
            return
        nef_widget = get_widget(renderer.nef)
        nef_widget.paint(state=state, module=renderer.nef)
