# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import numpy as np
import torch
import imgui
from wisp.framework.state import WispState
from wisp.models.nefs import SPCField
from wisp.models import Pipeline
from wisp.tracers import PackedSPCTracer
from wisp.renderer.gui import WidgetImgui
from wisp.renderer.core.api import add_to_scene_graph, remove_from_scene_graph


class WidgetSPCSelector(WidgetImgui):
    """ A custom widget which lets users browse SPC files, and select them to populate the scene graph. """

    def __init__(self):
        self.curr_file_idx = 0
        self.inited = False

    def create_pipeline(self, filename, device):
        # Load SPC content from file
        spc_fields = np.load(filename)

        # Convert to torch tensors on cuda device
        octree = torch.from_numpy(spc_fields['octree']).to(device)
        features = {k: torch.from_numpy(v).to(device) for k, v in spc_fields.items() if k in ('colors', 'normals')}

        neural_field = SPCField(
            octree=octree,
            features_dict=features,
            device=device
        )
        tracer = PackedSPCTracer()
        return Pipeline(neural_field, tracer)

    def paint(self, state: WispState, *args, **kwargs):
        """ Paint will be automatically called by the gui.
            Each widget included by the BrowseSPCApp will be automatically painted.
        """
        expanded, _ = imgui.collapsing_header("Object Browser", visible=True, flags=imgui.TREE_NODE_DEFAULT_OPEN)
        if expanded:
            available_files = state.extent['available_files']
            if len(available_files) == 0:
                if not self.inited:
                    print('Warning: No SPC files were found in input folder!')
                    self.inited = True
                return
            file_names = []

            for fpath in available_files:
                fnameext = os.path.basename(fpath)
                fname = os.path.splitext(fnameext)[0]
                file_names.append(fname)

            is_clicked, selected_file_idx = imgui.combo("Filename", self.curr_file_idx,
                                                        file_names)  # display your choices here
            if is_clicked or not self.inited:
                old_file_name = file_names[self.curr_file_idx]
                new_file_name = file_names[selected_file_idx]

                # Add new object to the scene graph
                # This will toggle a "marked as dirty" flag,
                # which forcees the render core to refresh the scene graph and load the new object
                device = state.renderer.device
                spc_pipeline = self.create_pipeline(available_files[selected_file_idx], device=device)
                add_to_scene_graph(state, name=new_file_name, obj=spc_pipeline)

                # Remove old object from scene graph, if it exists
                if old_file_name != new_file_name and old_file_name in state.graph.neural_pipelines:
                    remove_from_scene_graph(state, name=old_file_name)

                self.curr_file_idx = selected_file_idx
            self.inited = True
