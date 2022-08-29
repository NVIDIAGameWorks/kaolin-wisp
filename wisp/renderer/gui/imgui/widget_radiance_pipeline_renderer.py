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
from wisp.renderer.core.renderers import NeuralRadianceFieldPackedRenderer
from .widget_imgui import WidgetImgui
from .widget_radiance_pipeline import WidgetNeuralRadianceField
from .widget_property_editor import WidgetPropertyEditor


class WidgetNeuralRadianceFieldRenderer(WidgetImgui):
    names_to_widgets = dict(
        NeuralRadianceField=WidgetNeuralRadianceField
    )

    names_to_colors = {
        "Octree Grid": light_yellow,
        "Codebook Grid": light_cyan,
        "Triplanar Grid": light_pink,
        "Hash Grid": light_teal,
        "Unknown": gray
    }

    marcher_types = ['ray', 'voxel']

    def __init__(self):
        super().__init__()
        self.pipeline_widget = None
        self.properties_widget = WidgetPropertyEditor()

    def get_pipeline_widget(self, pipeline_name):
        if self.pipeline_widget is None:
            if pipeline_name in self.names_to_widgets:
                self.pipeline_widget = self.names_to_widgets[pipeline_name]()   
        return self.pipeline_widget

    def paint(self, state: WispState, renderer: NeuralRadianceFieldPackedRenderer = None, *args, **kwargs):
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
                MAX_SAMPLES = 128               # For general tracers
                MAX_SAMPLES_RAY_MODE = 512      # For 'ray' sampling mode

                # TODO (operel): Update the ## ids below with a unique object name to avoid imgui bug
                def _num_samples_property():
                    max_value = MAX_SAMPLES
                    if renderer.raymarch_type == 'ray':
                        max_value = MAX_SAMPLES_RAY_MODE
                    value = min(renderer.num_steps, max_value)
                    changed, value = imgui.core.slider_int(f"##samples_per_ray", value=value,
                                                           min_value=2, max_value=max_value)
                    if changed:
                        renderer.num_steps = value
                
                def _num_samples_movement_property():
                    max_value = MAX_SAMPLES
                    if renderer.raymarch_type == 'ray':
                        max_value = MAX_SAMPLES_RAY_MODE
                    value = min(renderer.num_steps_movement, max_value)
                    changed, value = imgui.core.slider_int(f"##samples_per_ray_movement", value=value,
                                                           min_value=2, max_value=max_value)
                    if changed:
                        renderer.num_steps_movement = value

                def _batch_size_property():
                    changed, value = imgui.core.slider_int("##batch_size", value=renderer.batch_size,
                                                           min_value=2**9, max_value=2**18)
                    if changed:
                        renderer.batch_size = value

                def _marcher_type_property():
                    selected_marcher_idx = self.marcher_types.index(renderer.raymarch_type)
                    changed, selected_marcher_idx = imgui.combo("##marcher_type",
                                                                selected_marcher_idx, self.marcher_types)
                    if changed:
                        new_marcher_mode = self.marcher_types[selected_marcher_idx]
                        if new_marcher_mode != 'ray':
                            renderer.num_steps = min(MAX_SAMPLES, renderer.num_steps)
                        renderer.raymarch_type = new_marcher_mode

                properties = {
                    "Ray Samples (static)": _num_samples_property,              # Samples per ray
                    "Ray Samples (movement)": _num_samples_movement_property,   # Samples per ray
                    "Batch Size (Rays)": _batch_size_property,
                    "Marcher Type": _marcher_type_property,
                    "Render Resolution (W x H)": f"{renderer.render_res_x} x {renderer.render_res_y}"
                }
                self.properties_widget.paint(state=state, properties=properties)
                imgui.tree_pop()

            if renderer.nef is not None:
                pipeline = renderer.nef
                pipeline_name = type(pipeline).__name__
                pipeline_widget = self.get_pipeline_widget(pipeline_name)
                if pipeline_widget is not None:
                    pipeline_widget.paint(state, pipeline)
