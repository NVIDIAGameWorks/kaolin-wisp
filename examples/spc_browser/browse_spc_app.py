# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import glob
from typing import List
from wisp.framework.state import WispState
from wisp.renderer.gui import WidgetImgui, WidgetInteractiveVisualizerProperties, WidgetGPUStats, WidgetSceneGraph
from wisp.renderer.app.wisp_app import WispApp
from widget_spc_selector import WidgetSPCSelector


class BrowseSPCApp(WispApp):
    """ An app for loading and visualizing Structured Point Cloud (SPC) content from kaolin """

    def __init__(self, wisp_state: WispState, window_name: str):
        super().__init__(wisp_state, window_name)

    def init_wisp_state(self, wisp_state: WispState) -> None:
        """ A hook for applications to initialize specific fields inside the wisp state object.
            This function is called before the entire renderer is constructed, hence initialized renderer fields can
            be defined to affect the behaviour of the renderer.
        """
        # Channels available to view over the canvas
        wisp_state.renderer.available_canvas_channels = ["RGB", "Depth"]
        wisp_state.renderer.selected_canvas_channel = "RGB"

        # For this app, we'll use only a world grid which resides in the 'xy' plane.
        # This attribute will signal the renderer that some camera controllers should align with this plane,
        # as well as what gizmos (e.g: reference grids, axes) to draw on the canvas.
        wisp_state.renderer.reference_grids = ['xy']

        # Custom app fields
        spc_dir = wisp_state.extent['dataset_path']
        available_files = sorted(glob.glob(spc_dir + "/*.npz"))
        wisp_state.extent['available_files'] = available_files  # store inside wisp state object for later

    def create_widgets(self) -> List[WidgetImgui]:
        """ Define the list of widgets the gui will display, in order. """
        return [
            WidgetGPUStats(),  # Current FPS, memory occupancy, GPU Model
            WidgetInteractiveVisualizerProperties(),  # Canvas dims, user camera controller & definitions
            WidgetSPCSelector(),  # Custom widget for selecting which SPC model to show
            WidgetSceneGraph()  # A scene graph tree of the entire hierarchy of objects in the scene
        ]

    def default_user_mode(self) -> str:
        """ Set the default camera controller mode.
            Possible choices: 'First Person View', 'Turntable', 'Trackball' """
        return "Trackball"
