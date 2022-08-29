# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import imgui
from wisp.core.colors import light_cyan, light_yellow, light_pink, light_teal, gray
from wisp.framework import WispState
from wisp.renderer.core.renderers import NeuralSDFPackedRenderer
from .widget_imgui import WidgetImgui
from .widget_sdf_pipeline import WidgetNeuralSDF
from .widget_property_editor import WidgetPropertyEditor


class WidgetNeuralSDFRenderer(WidgetImgui):
    #TODO(ttakikawa): Rename to SDF
    names_to_widgets = dict(
        NeuralSDF=WidgetNeuralSDF
    )

    names_to_colors = {
        "Octree": light_yellow,
        "Codebook Grid": light_cyan,
        "Triplanar Grid": light_pink,
        "Hash Grid": light_teal,
        "Unknown": gray
    }

    def __init__(self):
        super().__init__()
        self.pipeline_widget = None
        self.properties_widget = WidgetPropertyEditor()

    def get_pipeline_widget(self, pipeline_name):
        if self.pipeline_widget is None:
            if pipeline_name in self.names_to_widgets:
                self.pipeline_widget = self.names_to_widgets[pipeline_name]()   
        return self.pipeline_widget

    def paint(self, state: WispState, renderer: NeuralSDFPackedRenderer = None, *args, **kwargs):
        if renderer is not None:
            imgui.text(f"Features Structure: ")
            imgui.same_line()
            feature_struct_name = renderer.features_structure()
            feature_struct_color = self.names_to_colors.get(feature_struct_name, gray)
            imgui.text_colored(f"{feature_struct_name}", *feature_struct_color)
            imgui.text(f"Acceleration Structure: ")
            imgui.same_line()
            imgui.text(f"{renderer.acceleration_structure()}")

            if imgui.tree_node("Tracer", imgui.TREE_NODE_DEFAULT_OPEN):

                def _num_samples_property():
                    changed, value = imgui.core.slider_int(f"##steps_per_ray", value=renderer.samples_per_ray,
                                                           min_value=1, max_value=256)
                    renderer.samples_per_ray = value

                def _batch_size_property():
                    changed, value = imgui.core.slider_int("##batch_size", value=renderer.batch_size,
                                                           min_value=2**9, max_value=2**15)
                    renderer.batch_size = value

                properties = {
                    "Steps per ray": _num_samples_property,
                    # "Batch Size (Rays)": _batch_size_property,
                    "Render Resolution (W x H)": f"{renderer.render_res_x} x {renderer.render_res_y}",
                }
                self.properties_widget.paint(state=state, properties=properties)
                imgui.tree_pop()

            if renderer.nef is not None:
                sdf = renderer.nef
                pipeline_name = type(sdf).__name__
                pipeline_widget = self.get_pipeline_widget(pipeline_name)
                if pipeline_widget is not None:
                    pipeline_widget.paint(state, sdf)
