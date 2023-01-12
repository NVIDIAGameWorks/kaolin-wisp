# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import imgui
from wisp.framework import WispState
from wisp.renderer.core.renderers import NeuralSDFPackedRenderer
from .widget_imgui import WidgetImgui, get_widget, widget
from .widget_property_editor import WidgetPropertyEditor


@widget(NeuralSDFPackedRenderer)
class WidgetNeuralSDFRenderer(WidgetImgui):

    def __init__(self):
        super().__init__()
        self.properties_widget = WidgetPropertyEditor()

    def paint_tracer(self, state: WispState, renderer: NeuralSDFPackedRenderer):
        if imgui.tree_node("Tracer", imgui.TREE_NODE_DEFAULT_OPEN):
            def _num_samples_property():
                changed, value = imgui.core.slider_int(f"##steps_per_ray", value=renderer.samples_per_ray,
                                                       min_value=1, max_value=256)
                renderer.samples_per_ray = value

            def _batch_size_property():
                changed, value = imgui.core.slider_int("##batch_size", value=renderer.batch_size,
                                                       min_value=2 ** 9, max_value=2 ** 15)
                renderer.batch_size = value

            properties = {
                "Steps per ray": _num_samples_property,
                # "Batch Size (Rays)": _batch_size_property,
                "Render Resolution (W x H)": f"{renderer.render_res_x} x {renderer.render_res_y}",
            }
            self.properties_widget.paint(state=state, properties=properties)
            imgui.tree_pop()

    def paint(self, state: WispState, renderer: NeuralSDFPackedRenderer = None, *args, **kwargs):
        if renderer is None:
            return
        self.paint_tracer(state, renderer)
        nef_widget = get_widget(renderer.nef)
        nef_widget.paint(state=state, module=renderer.nef)
