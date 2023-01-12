# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import imgui
from wisp.core import WispModule
from wisp.accelstructs import BaseAS
from wisp.models.grids import BLASGrid
from wisp.models.nefs import BaseNeuralField
from wisp.models.decoders import BasicDecoder
from wisp.tracers import BaseTracer
from wisp.models.embedders import PositionalEmbedder
from wisp.framework import WispState
from .widget_imgui import WidgetImgui, widget, get_widget
from .widget_property_editor import WidgetPropertyEditor


@widget(WispModule)
class WidgetWispModule(WidgetImgui):

    def __init__(self):
        super().__init__()
        self.properties_widget = WidgetPropertyEditor()

    def get_type_title(self, module: WispModule):
        if isinstance(module, BaseAS):
            return "Acceleration Structure"
        elif isinstance(module, BLASGrid):
            return "Grid"
        elif isinstance(module, BaseNeuralField):
            return "Neural Field"
        elif isinstance(module, BasicDecoder):          # TODO (operel): interface decoders
            return "Decoder"
        elif isinstance(module, PositionalEmbedder):    # TODO (operel): interface embedders
            return "Embedder"
        elif isinstance(module, BaseTracer):
            return "Tracer"
        else:
            return "Module"

    def paint(self, state: WispState, module: WispModule = None, field_name=None, *args, **kwargs):
        if module is None:
            return

        type_title = self.get_type_title(module)
        section_title = field_name if field_name is not None else type_title
        if imgui.tree_node(section_title):
            imgui.text(f"Type: {module.name()}")

            properties = module.public_properties()
            recursive_properties = {k: p for k, p in properties.items() if isinstance(p, WispModule)}
            for recursive_field_name, recursive_property in recursive_properties.items():
                child_widget = get_widget(recursive_property)
                child_widget.paint(state=state, module=recursive_property, field_name=recursive_field_name)

            table_properties = {k: p for k, p in properties.items() if not isinstance(p, WispModule)}
            self.properties_widget.paint(state=state, properties=table_properties)
            imgui.tree_pop()
