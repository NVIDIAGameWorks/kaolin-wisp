# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import imgui
from wisp.core import WispModule
from wisp.framework import WispState
from wisp.accelstructs import BaseAS
from .widget_imgui import WidgetImgui, widget, get_widget
from .widget_property_editor import WidgetPropertyEditor


@widget(BaseAS)
class WidgetAccelStruct(WidgetImgui):

    def __init__(self):
        super().__init__()
        self.properties_widget = WidgetPropertyEditor()

    def paint(self, state: WispState, module: BaseAS = None, field_name = None, *args, **kwargs):
        if module is None:
            return

        section_title = field_name if field_name is not None else "Acceleration Structure"
        if imgui.tree_node(section_title):
            imgui.text(f"Type: {module.name()}")

            imgui.text(f"Sparsity (per LOD):")
            occupancy_hist = [o / c for o, c in zip(module.occupancy(), module.capacity())]
            width, height = imgui.get_content_region_available()
            imgui.plot_histogram(label="##accelstruct", values=np.array(occupancy_hist, dtype=np.float32),
                                 graph_size=(width, 20), scale_min=0.0, scale_max=1.0)

            properties = module.public_properties()
            table_properties = {k: p for k, p in properties.items() if not isinstance(p, WispModule)}
            recursive_properties = {k: p for k, p in properties.items() if isinstance(p, WispModule)}
            self.properties_widget.paint(state=state, properties=table_properties)

            for recursive_field_name, recursive_property in recursive_properties.items():
                child_widget = get_widget(recursive_property)
                child_widget.paint(state=state, module=recursive_property, field_name=recursive_field_name)
            imgui.tree_pop()
