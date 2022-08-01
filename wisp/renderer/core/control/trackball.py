# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
import copy
import torch
import wisp.framework.state as state
from wisp.renderer.core.control.camera_controller_mode import CameraControlMode
from wisp.renderer.core.control.io import WispMouseButton

# A trackball camera, allowing free orbit movement in all axes.
# Also known as Arcball, or "Orbit" camera
# Original algorithm from:
# * ARCBALL: A User Interface for Specifying Three-Dimensional Orientation Using a Mouse,
#   Shoemake, K, Proceedings of the conference on Graphics interface 1992
#
# * Animating rotation with quaternion curves, Shoemake, K, SIGGRAPH 1985

def quat_mul(Q1, Q2):
    return torch.tensor([Q1[0] * Q2[3] + Q1[3] * Q2[0] - Q1[2] * Q2[1] + Q1[1] * Q2[2],
                         Q1[1] * Q2[3] + Q1[2] * Q2[0] + Q1[3] * Q2[1] - Q1[0] * Q2[2],
                         Q1[2] * Q2[3] - Q1[1] * Q2[0] + Q1[0] * Q2[1] + Q1[3] * Q2[2],
                         Q1[3] * Q2[3] - Q1[0] * Q2[0] - Q1[1] * Q2[1] - Q1[2] * Q2[2]])


def quat_matrix(q): # True only for unit quaternions
    xx = q[0] * q[0]
    xy = q[0] * q[1]
    xz = q[0] * q[2]
    xw = q[0] * q[3]
    yy = q[1] * q[1]
    yz = q[1] * q[2]
    yw = q[1] * q[3]
    zz = q[2] * q[2]
    zw = q[2] * q[3]
    ww = q[3] * q[3]
    return torch.tensor([[ww + xx - yy - zz, 2.0 * (xy - zw), 2.0 * (xz + yw), 0.0],
                         [2.0 * (xy + zw), ww - xx + yy - zz, 2.0 * (yz - xw), 0.0],
                         [2.0 * (xz - yw), 2.0 * (yz + xw), ww - xx - yy + zz, 0.0],
                         [0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)


class TrackballCameraMode(CameraControlMode):

    def __init__(self, render_core, wisp_state: state.WispState):
        super().__init__(render_core, wisp_state)
        self.radius = 1.0
        self.tb_scale = 1.1
        self.sensitivity = 1.0
        self.reset_center_of_focus(reset_radius=True)

        self.q = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.q0 = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.q1 = None
        self.v0 = None
        self.v1 = None

        self.initial_camera = None
        self.planes_forbidden_zooming_through = self.state.renderer.reference_grids

    @classmethod
    def name(cls):
        return "Trackball"
 
    def reset_center_of_focus(self, reset_radius=False):
        pos = self.camera.cam_pos().squeeze()
        forward_axis = self.camera.cam_forward().squeeze()
        forward_axis /= forward_axis.norm()
        if reset_radius:
            self.radius = torch.dot(pos, forward_axis)
        self.focus_at = pos - self.radius * forward_axis   

    def mouse2vector(self, mx, my):
        """ Converts mouse position in screen coordinates:
            [0,0] at top left of screen to [width, height] at bottom-right
            to coordinates on arcball
        """
        # First we convert the mouse coordinates to normalized device coordinates, multiplied by the arcball radius.
        # v is a vector of (x,y,z), and will contain coordinates projected to the arcball surface
        # TODO: Is x and y rotated inversely?
        half_width = 0.5 * self.camera.width
        half_height = 0.5 * self.camera.height
        v = torch.tensor([half_width - float(mx), float(my) - half_height, 0.0])
        normalization_factor = half_height if half_width >= half_height else half_width
        v *= self.tb_scale / float(normalization_factor)

        # v is now in Normalized Device Coordinates
        # Next we need to calculate the z coordinate, which is currently set to 0

        # xy_power = x^2 + y^2
        xy_power = torch.pow(v, 2).sum()

        if xy_power < 1.0:
            v[2] = -torch.sqrt(1.0 - xy_power)   # z = -sqrt(R - x^2 - y^2)
        else:
            v /= torch.sqrt(xy_power)   # x = x/sqrt(x^2 + y^2) ; y = y/sqrt(x^2 + y^2) ; z = 0.0

        return v

    def handle_mouse_press(self, x, y, button):
        self.stop_all_current_interactions()
        if button == WispMouseButton.LEFT_BUTTON:
            self.start_interaction('trackball_rotate')
            self.initial_camera = copy.deepcopy(self.camera)
            self.v0 = self.mouse2vector(x, y)
        elif button == WispMouseButton.MIDDLE_BUTTON:
            super().handle_mouse_press(x, y, button)

    def handle_mouse_release(self, x, y, button):
        if self.has_interaction('trackball_rotate'):
            self.end_interaction()
            self.q0 = torch.tensor([0.0, 0.0, 0.0, 1.0])
            self.initial_camera = None
        else:
            super().handle_mouse_release(x, y, button)

    def handle_mouse_drag(self, x, y, dx, dy, button):
        if not self.has_interaction('trackball_rotate'):
            super().handle_mouse_drag(x, y, dx, dy, button)
        else:
            camera = self.camera

            if torch.allclose(self.v0, torch.zeros_like(self.v0)):
                self.v0 = self.mouse2vector(x, y)   # Start position

            self.v1 = self.mouse2vector(x, y)       # Current position

            # find quaterion for previous to current frame rotation
            axis = torch.cross(self.v1, self.v0)
            angle = -torch.dot(self.v1, self.v0).unsqueeze(0)
            angle *= self.sensitivity
            self.q1 = torch.cat((axis, angle))

            # Apply rotation to starting rotation quaternion
            self.q = quat_mul(self.q1, self.q0)

            # To rotation matrix
            Q = quat_matrix(self.q)
            Q = Q.to(camera.device)

            # find length of vector from eye to focus at point
            pos = self.initial_camera.cam_pos().reshape(-1)
            vec = self.focus_at - pos
            length = torch.sqrt(torch.dot(vec, vec))

            # create translation in z direction to/from focua at point (in camera space)
            T = torch.eye(4, device=camera.device, dtype=camera.dtype).repeat(1, 1)
            T[2, 3] = length
            Tinv = torch.eye(4, device=camera.device, dtype=camera.dtype).repeat(1, 1)
            Tinv[2, 3] = -length

            view_matrix = self.initial_camera.view_matrix()[0]   # Unbatch

            # apply transforms
            view_matrix = Tinv @ Q @ T @ view_matrix

            # update extrinsic matrix
            camera.update(view_matrix)

