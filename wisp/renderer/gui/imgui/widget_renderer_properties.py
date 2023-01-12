# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import imgui
from .widget_imgui import WidgetImgui, widget
from .widget_cameras import WidgetCameraProperties
from wisp.core.colors import black, white, dark_gray
from wisp.framework import WispState, InteractiveRendererState
from wisp.renderer.core.control import FirstPersonCameraMode, TrackballCameraMode, TurntableCameraMode


@widget(InteractiveRendererState)
class WidgetInteractiveVisualizerProperties(WidgetImgui):
    def __init__(self):
        super().__init__()
        self.camera_properties = WidgetCameraProperties(camera_id="User Camera")

        self.available_camera_modes = [FirstPersonCameraMode, TrackballCameraMode, TurntableCameraMode]
        self.available_camera_mode_names = [mode.name() for mode in self.available_camera_modes]

    def paint(self, state: WispState, *args, **kwargs):
        expanded, _ = imgui.collapsing_header("Renderer", visible=True, flags=imgui.TREE_NODE_DEFAULT_OPEN)
        if expanded:
            imgui.text(f"Canvas Size (W,H): "
                       f"{state.renderer.canvas_width} x {state.renderer.canvas_height}")

            bg_btn_width = bg_btn_height = 15
            imgui.text("Background Color: ")
            imgui.same_line()
            bg_buttons = {
                'Black': black,
                'Dark Gray': dark_gray,
                'White': white,
            }
            for btn_caption, btn_color in bg_buttons.items():
                imgui.same_line()
                if imgui.color_button(btn_caption, *btn_color, 1.0, 0, bg_btn_width, bg_btn_height):
                    state.renderer.clear_color_value = btn_color

            available_canvas_channels = state.renderer.available_canvas_channels
            selected_canvas_channel = state.renderer.selected_canvas_channel
            if selected_canvas_channel is not None and available_canvas_channels is not None and \
                    selected_canvas_channel in available_canvas_channels:
                curr_channel_idx = available_canvas_channels.index(selected_canvas_channel)
                clicked, new_channel_idx = imgui.combo("Channel", curr_channel_idx, available_canvas_channels)
                if new_channel_idx != curr_channel_idx:
                    state.renderer.selected_canvas_channel = available_canvas_channels[new_channel_idx]

            cam_mode = state.renderer.cam_controller
            if cam_mode is not None:
                curr_mode_idx = self.available_camera_mode_names.index(cam_mode.name())
                clicked, new_mode_idx = imgui.combo("Controller", curr_mode_idx, self.available_camera_mode_names)
                if new_mode_idx != curr_mode_idx:  # Update state with new mode if changed
                    new_mode_cls = self.available_camera_modes[new_mode_idx]
                    state.renderer.cam_controller = new_mode_cls

            if imgui.tree_node("User Camera", imgui.TREE_NODE_DEFAULT_OPEN):
                self.camera_properties.paint(state, camera=state.renderer.selected_camera)
                imgui.tree_pop()
