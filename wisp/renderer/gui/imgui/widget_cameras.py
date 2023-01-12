# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import copy

import imgui
from typing import Optional
from kaolin.render.camera import Camera
from wisp.framework import WispState
from .widget_imgui import WidgetImgui, widget, get_widget
from .widget_property_editor import WidgetPropertyEditor


@widget(Camera)
class WidgetCameraProperties(WidgetImgui):
    def __init__(self, camera_id: Optional[str] = None):
        super().__init__()

        self.camera_id = camera_id
        self.available_cameras = ['Perspective', 'Orthographic']
        self.intrinsics_to_selection_idx = dict(
            PinholeIntrinsics=self.available_cameras.index('Perspective'),
            OrthographicIntrinsics=self.available_cameras.index('Orthographic')
        )
        self.properties_editor = WidgetPropertyEditor()

    def paint(self, state: WispState, camera: Camera = None, *args, **kwargs):

        if camera is None:
            camera = state.renderer.selected_camera

        is_editable = camera == state.renderer.selected_camera

        if camera != state.renderer.selected_camera:
            button_width = 100
            if imgui.button("Set View", width=button_width):
                prev_selected_cam = state.renderer.selected_camera
                device = prev_selected_cam.device
                dtype = prev_selected_cam.dtype
                state.renderer.selected_camera = copy.deepcopy(camera).to(device, dtype)

        if imgui.tree_node("Intrinsics", imgui.TREE_NODE_DEFAULT_OPEN):
            cam_type = type(camera.intrinsics).__name__
            selected_cam_idx = self.intrinsics_to_selection_idx[cam_type]
            clicked, new_cam_idx = imgui.combo("Lens", selected_cam_idx, self.available_cameras)
            if is_editable and selected_cam_idx != new_cam_idx:
                camera = copy.deepcopy(camera)  # Camera may change, but let gui finish drawing the previous cam info
                state.renderer.selected_camera_lens = self.available_cameras[new_cam_idx]

            properties = None
            if cam_type == 'PinholeIntrinsics':
                properties = self._pinhole_properties(camera, is_editable)
            elif cam_type == 'OrthographicIntrinsics':
                properties = self._ortho_properties(camera, is_editable)
            if properties is not None:
                self.properties_editor.paint(state=state, properties=properties)
            imgui.tree_pop()

        if imgui.tree_node("Extrinsics", imgui.TREE_NODE_DEFAULT_OPEN):
            flags = imgui.INPUT_TEXT_ALWAYS_INSERT_MODE | imgui.INPUT_TEXT_CHARS_DECIMAL
            rchanged, rvalues = imgui.input_float3("Right", *camera.cam_right()[0].squeeze().cpu().numpy(),
                                                   flags=flags)
            uchanged, uvalues = imgui.input_float3("Up", *camera.cam_up()[0].squeeze().cpu().numpy(),
                                                   flags=flags)
            fchanged, fvalues = imgui.input_float3("Forward", *camera.cam_forward()[0].squeeze().cpu().numpy(),
                                                   flags=flags)
            tchanged, tvalues = imgui.input_float3("Position", *camera.cam_pos()[0].squeeze().cpu().numpy(),
                                                   flags=flags)

            if rchanged or uchanged or fchanged or tchanged:
                # TODO (operel) - make sure mem layout is ok
                # TODO (operel) - don't allow to edit training cameras because that would ruin
                updated_R = camera[0].R.new_tensor([rvalues, uvalues, fvalues]).transpose(-1,-2)
                camera[0].R = updated_R
            if tchanged:
                updated_t = camera[0].t.new_tensor([tvalues]).transpose(-1,-2)
                camera[0].t = updated_t
            imgui.tree_pop()

    def _pinhole_properties(self, camera, is_editable):
        def _x0_property():
            x0_value = camera.x0.item()
            changed, new_x0_value = imgui.core.slider_int("##camera_x0",
                                                          value=x0_value, min_value=0, max_value=camera.width)
            if changed and x0_value != new_x0_value:
                camera.x0 = new_x0_value

        def _y0_property():
            y0_value = camera.y0.item()
            changed, new_y0_value = imgui.core.slider_int("##camera_y0",
                                                          value=y0_value, min_value=0, max_value=camera.height)
            if changed and y0_value != new_y0_value:
                camera.y0 = new_y0_value

        fov_x_value = camera.fov_x.item()
        fov_y_value = camera.fov_y.item()
        fov_ratio = fov_x_value / fov_y_value

        near_value = camera.near
        def _near_property():
            changed, new_near_value = imgui.core.slider_float("##camera_near",
                                                              value=near_value, min_value=1e-2, max_value=20.0)
            if changed and near_value != new_near_value:
                camera.near = new_near_value

        far_value = camera.far
        def _far_property():
            changed, new_far_value = imgui.core.slider_float("##camera_far",
                                                              value=far_value, min_value=1.0, max_value=100.0)
            if changed and far_value != new_far_value:
                camera.far = new_far_value


        def _fov_x_property():
            changed, new_fov_x_value = imgui.core.slider_float("##camera_fov_x",
                                                               value=fov_x_value, min_value=1.0, max_value=120.0)
            if changed and fov_x_value != new_fov_x_value:
                camera.fov_x = new_fov_x_value
                camera.fov_y = new_fov_x_value / fov_ratio

        def _fov_y_property():
            changed, new_fov_y_value = imgui.core.slider_float("##camera_fov_y",
                                                               value=fov_y_value, min_value=1.0, max_value=120.0)
            if changed and fov_y_value != new_fov_y_value:
                camera.fov_y = new_fov_y_value
                camera.fov_x = new_fov_y_value * fov_ratio

        properties = {
            'Near Clip Plane': _near_property if is_editable else camera.near,
            'Far Clip Plane': _far_property if is_editable else camera.far,
            'Field of view (H)': _fov_x_property if is_editable else camera.fov_x.item(),
            'Field of view (V)': _fov_y_property if is_editable else camera.fov_y.item(),
            'Principal Point x0': _x0_property if is_editable else camera.x0.item(),
            'Principal Point y0': _y0_property if is_editable else camera.y0.item(),
            'Focal Length X': camera.focal_x.item(),
            'Focal Length Y': camera.focal_y.item(),
        }
        return properties

    def _ortho_properties(self, camera, is_editable):
        def _fov_distance():
            fov_distance_value = camera.fov_distance.item()
            MAX_ORTHO_ZOOMOUT = 75  # In distance units
            max_clipping_plane = min(max(camera.far, camera.near), MAX_ORTHO_ZOOMOUT)
            changed, new_fov_dist_value = imgui.core.slider_float("##camera_fov_distance",
                                                                  value=fov_distance_value,
                                                                  min_value=1e-4, max_value=max_clipping_plane)
            if changed and fov_distance_value != new_fov_dist_value:
                camera.fov_distance = new_fov_dist_value

        properties = {
            'FOV distance': _fov_distance if is_editable else camera.fov_distance.item(),
        }
        return properties
