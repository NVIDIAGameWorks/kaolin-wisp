# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
import numpy as np
import wisp.framework.state as state
from wisp.renderer.core.control.camera_controller_mode import CameraControlMode
from wisp.renderer.core.control.io import WispMouseButton


class FirstPersonCameraMode(CameraControlMode):

    def __init__(self, render_core, wisp_state: state.WispState):
        super().__init__(render_core, wisp_state)
        self._rotation_sensitivity = 0.2

    @classmethod
    def name(cls):
        return "First Person View"

    def handle_mouse_drag(self, x, y, dx, dy, button):
        if self.has_interaction('fps_rotate'):
            camera = self.render_core.camera
            w, h = camera.width, camera.height
            max_rot = 2 * self.PI * self._rotation_sensitivity
            full_dist = np.minimum(h, w)
            yaw = -max_rot * dx / full_dist  # Rotates camera in right-forward plane
            pitch = -max_rot * dy / full_dist  # Rotates camera in up-forward plane
            self.render_core.camera.rotate(yaw, pitch)
        else:
            super().handle_mouse_drag(x, y, dx, dy, button)

    def handle_mouse_press(self, x, y, button):
        self.stop_all_current_interactions()
        if button == WispMouseButton.LEFT_BUTTON:
            self.start_interaction('fps_rotate')
        elif button == WispMouseButton.MIDDLE_BUTTON:
            super().handle_mouse_press(x, y, button)

    def handle_mouse_release(self, x, y, button):
        if self.has_interaction('fps_rotate'):
            self.end_interaction()
        else:
            super().handle_mouse_release(x, y, button)
