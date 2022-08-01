# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
import numpy as np
import torch
import wisp.framework.state as state
from wisp.renderer.core.control.camera_controller_mode import CameraControlMode
from wisp.renderer.core.control.io import WispMouseButton


# A turntable camera with 2 degrees of freedom.
# This camera controller is more restrictive, but maintains the "horizon" orientation to
# allow for a more intuitive interface.

class TurntableCameraMode(CameraControlMode):

    def __init__(self, render_core, wisp_state: state.WispState):
        super().__init__(render_core, wisp_state)
        self.sensitivity = 0.95
        self.reset_center_of_focus(reset_radius=True)

        reference_grids = self.state.renderer.reference_grids
        self.planes_forbidden_zooming_through = reference_grids
        self.reference_plane = reference_grids[0] if len(reference_grids) > 0 else "xy"

    @classmethod
    def name(cls):
        return "Turntable"

    def reset_center_of_focus(self, reset_radius=False):
        pos = self.camera.cam_pos().squeeze()
        forward_axis = self.camera.cam_forward().squeeze()
        forward_axis /= forward_axis.norm()
        if reset_radius:
            self.radius = torch.dot(pos, forward_axis)
        self.focus_at = pos - self.radius * forward_axis

    def yaw_camera_in_world_space(self, camera, yaw):
        translate = torch.eye(4, device=camera.device, dtype=camera.dtype).repeat(1, 1)
        translate[:3, 3] = self.focus_at
        retranslate = torch.eye(4, device=camera.device, dtype=camera.dtype).repeat(1, 1)
        translate[:3, 3] = -self.focus_at

        rot_yaw = torch.eye(4, device=camera.device, dtype=camera.dtype).repeat(1, 1)
        rot_yaw[0, 0] = torch.cos(yaw)
        rot_yaw[0, 2] = -torch.sin(yaw)
        rot_yaw[2, 0] = torch.sin(yaw)
        rot_yaw[2, 2] = torch.cos(yaw)
        view_matrix = camera.view_matrix()[0]   # Unbatch
        # Translate world center to focus point, rotate, and translate back
        view_matrix = view_matrix @ retranslate @ rot_yaw @ translate
        view_matrix = view_matrix.unsqueeze(0)
        camera.update(view_matrix)

    def pitch_camera_in_world_space(self, camera, pitch):
        translate = torch.eye(4, device=camera.device, dtype=camera.dtype).repeat(1, 1)
        translate[:3, 3] = self.focus_at
        retranslate = torch.eye(4, device=camera.device, dtype=camera.dtype).repeat(1, 1)
        translate[:3, 3] = -self.focus_at

        rot_pitch = torch.eye(4, device=camera.device, dtype=camera.dtype).repeat(1, 1)
        rot_pitch[1, 1] = torch.cos(pitch)
        rot_pitch[1, 2] = torch.sin(pitch)
        rot_pitch[2, 1] = -torch.sin(pitch)
        rot_pitch[2, 2] = torch.cos(pitch)
        view_matrix = camera.view_matrix()[0]   # Unbatch
        # Translate world center to focus point, rotate, and translate back
        view_matrix = view_matrix @ retranslate @ rot_pitch @ translate
        view_matrix = view_matrix.unsqueeze(0)
        camera.update(view_matrix)

    def roll_camera_in_world_space(self, camera, roll):
        translate = torch.eye(4, device=camera.device, dtype=camera.dtype).repeat(1, 1)
        translate[:3, 3] = self.focus_at
        retranslate = torch.eye(4, device=camera.device, dtype=camera.dtype).repeat(1, 1)
        translate[:3, 3] = -self.focus_at

        rot_pitch = torch.eye(4, device=camera.device, dtype=camera.dtype).repeat(1, 1)
        rot_pitch[0, 0] = torch.cos(roll)
        rot_pitch[0, 1] = -torch.sin(roll)
        rot_pitch[1, 0] = torch.sin(roll)
        rot_pitch[1, 1] = torch.cos(roll)
        view_matrix = camera.view_matrix()[0]   # Unbatch
        # Translate world center to focus point, rotate, and translate back
        view_matrix = view_matrix @ retranslate @ rot_pitch @ translate
        view_matrix = view_matrix.unsqueeze(0)
        camera.update(view_matrix)

    def handle_mouse_drag(self, x, y, dx, dy, button):
        if self.has_interaction('turntable_rotate'):
            camera = self.camera
            w, h = camera.width, camera.height
            max_rot = 2 * self.PI * self.sensitivity
            full_dist = np.minimum(h, w)

            in_plane_amount = -max_rot * dx / full_dist  # Rotates camera in normal direction to plane (world space)
            pitch = -max_rot * dy / full_dist            # Rotates camera in up-forward direction (camera space)
            camera.t = torch.zeros(3, dtype=camera.dtype, device=camera.device)
            camera.rotate(pitch=pitch)

            if self.reference_plane == 'xz':
                self.yaw_camera_in_world_space(camera, in_plane_amount)
            elif self.reference_plane == 'xy':
                self.roll_camera_in_world_space(camera, in_plane_amount)
            elif self.reference_plane == 'yz':
                self.pitch_camera_in_world_space(camera, in_plane_amount)

            backward_dir = camera.cam_forward().squeeze() / camera.cam_forward().norm()
            camera.translate(self.radius * backward_dir)
        else:
            super().handle_mouse_drag(x, y, dx, dy, button)

    def handle_mouse_press(self, x, y, button):
        self.stop_all_current_interactions()
        if button == WispMouseButton.LEFT_BUTTON:
            self.start_interaction('turntable_rotate')
        elif button == WispMouseButton.MIDDLE_BUTTON:
            super().handle_mouse_press(x, y, button)

    def handle_mouse_release(self, x, y, button):
        if self.has_interaction('turntable_rotate'):
            self.end_interaction()
        else:
            super().handle_mouse_release(x, y, button)

    def end_pan(self):
        last_interaction = self.get_last_interaction_started()
        if last_interaction is not None:
            self.reset_center_of_focus(reset_radius=True)
        else:
            self.reset_center_of_focus(reset_radius=False)
        super().end_pan()
