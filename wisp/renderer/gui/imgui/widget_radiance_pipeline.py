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
        self.grid_widget = None
        self.properties_widget = WidgetPropertyEditor()

    def get_grid_widget(self, grid_name):
        if self.grid_widget is None:
            if grid_name in self.names_to_widgets:
                self.grid_widget = self.names_to_widgets[grid_name]()
        return self.grid_widget

    def paint(self, state: WispState, nef: NeuralRadianceField = None, *args, **kwargs):
        if nef is not None:
            if nef.grid is not None:
                # TODO (operel): Separate this into acceleration / features previously ("Acceleration Structure")
                if imgui.tree_node("Grid Details", imgui.TREE_NODE_DEFAULT_OPEN):

                    grid = nef.grid
                    grid_name = type(grid).__name__
                    grid_widget = self.get_grid_widget(grid_name)
                    imgui.text(f"Type: {grid_name}")
                    if grid_widget is not None:
                        grid_widget.paint(state, grid)
                    imgui.tree_pop()

            properties = {}
            if nef.pos_embed_dim is not None:
                properties["Positional enc. dims"] = nef.pos_embed_dim
            if nef.pos_embed_dim is not None:
                properties["View direction enc. dims"] = nef.view_embed_dim
            if len(properties) > 0:
                if imgui.tree_node("Decoder", imgui.TREE_NODE_DEFAULT_OPEN):
                    self.properties_widget.paint(state, properties=properties)
                    imgui.tree_pop()
