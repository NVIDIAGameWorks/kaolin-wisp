# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import imgui
from wisp.framework import WispState
from wisp.models.nefs.nerf import NeuralRadianceField
from .widget_imgui import WidgetImgui
from .widget_octree_grid import WidgetOctreeGrid
from .widget_dictionary_octree_grid import WidgetCodebookOctreeGrid
from .widget_triplanar_grid import WidgetTriplanarGrid
from .widget_property_editor import WidgetPropertyEditor


class WidgetNeuralRadianceField(WidgetImgui):
    names_to_widgets = dict(
        OctreeGrid=WidgetOctreeGrid,
        CodebookOctreeGrid=WidgetCodebookOctreeGrid,
        TriplanarGrid=WidgetTriplanarGrid,
    )

    def __init__(self):
        super().__init__()
        self.accel_widget = None
        self.properties_widget = WidgetPropertyEditor()

    def get_acceleration_structure_widget(self, acceleration_structure_name):
        if self.accel_widget is None:
            if acceleration_structure_name in self.names_to_widgets:
                self.accel_widets = self.names_to_widgets[acceleration_structure_name]()
        return self.accel_widget

    def paint(self, state: WispState, pipeline: NeuralRadianceField = None, *args, **kwargs):
        if pipeline is not None:
            if pipeline.grid is not None:
                # TODO (operel): Separate this into acceleration / features previously ("Acceleration Structure")
                if imgui.tree_node("Grid Details", imgui.TREE_NODE_DEFAULT_OPEN):

                    acceleration_structure = pipeline.grid
                    acceleration_structure_name = type(acceleration_structure).__name__
                    accel_widget = self.get_acceleration_structure_widget(acceleration_structure_name)
                    imgui.text(f"Type: {acceleration_structure_name}")
                    if accel_widget is not None:
                        accel_widget.paint(state, acceleration_structure)
                    imgui.tree_pop()

            properties = {}
            if pipeline.pos_embed_dim is not None:
                properties["Positional enc. dims"] = pipeline.pos_embed_dim
            if pipeline.pos_embed_dim is not None:
                properties["View direction enc. dims"] = pipeline.view_embed_dim
            if len(properties) > 0:
                if imgui.tree_node("Decoder", imgui.TREE_NODE_DEFAULT_OPEN):
                    self.properties_widget.paint(state, properties=properties)
                    imgui.tree_pop()
