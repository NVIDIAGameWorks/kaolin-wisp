# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import torch
import wisp.framework.state as state
from wisp.renderer.core.control.io import WispKey


class CameraControlMode(ABC):
    def __init__(self, render_core, wisp_state: state.WispState):
        self.render_core = render_core
        self.state = wisp_state

        self.PI = torch.tensor(np.pi, device=render_core.device)
        self.planes_forbidden_zooming_through = []

        # ----------------------------------
        # Sensitivity & velocity parameters
        # ----------------------------------
        # The controller uses kinematics for smooth FPS considerate transitions
        # Pan operations with the keyboard:
        self._key_pan_initial_velocity = 0.8           # start with this v0 velocity
        self._key_pan_deacceleration = 3.2             # deaccelerate with this acceleration factor a ~ 4*v0
        self._key_pan_distance_weight = 0.5            # pan amount is distance dependant, weighted by this factor

        # Pan operations with the mouse:
        self._mouse_pan_distance_weight = 0.002        # pan amount is distance dependant, weighted by this factor

        # Zoom operations with the mouse:
        self._mouse_zoom_initial_velocity = 10.0       # start with this v0 velocity
        self._mouse_zoom_deacceleration = 40.0         # deaccelerate with this acceleration factor a ~ 4*v0
        self._zoom_persp_distance_weight = 0.25        # zoom is cam-pos dependant, weighted by this factor
        self._zoom_ortho_distance_weight = 0.2         # zoom is fov-distance dependant, weighted by this factor
        self._zoom_ortho_fov_dist_range = (1e-4, 1e2)  # zoom in ortho mode is limited to this fov-distance range

        # Current state
        self._current_pan_velocity = 0
        self._current_pan_direction = None
        self._remaining_pan_time = 0

        self.interactions_stack = list()

    @classmethod
    @abstractmethod
    def name(cls):
        raise NotImplementedError("Camera modes should state a descriptive name")

    def handle_timer_tick(self, dt):
        self.progress_pan(dt)

    def handle_mouse_scroll(self, x, y, dx, dy):
        """ The mouse wheel was scrolled by (dx,dy). """
        self.stop_all_current_interactions()
        if dy < 0:
            self.start_pan(pan_direction='forward',
                           velocity=self._mouse_zoom_initial_velocity, deaccelaration=self._mouse_zoom_deacceleration)
        else:
            self.start_pan(pan_direction='backward',
                           velocity=self._mouse_zoom_initial_velocity, deaccelaration=self._mouse_zoom_deacceleration)

    def handle_key_press(self, symbol, modifiers):
        self.stop_all_current_interactions()
        if symbol == WispKey.LEFT:
            self.start_pan(pan_direction='left',
                           velocity=self._key_pan_initial_velocity, deaccelaration=self._key_pan_deacceleration)
        elif symbol == WispKey.RIGHT:
            self.start_pan(pan_direction='right',
                           velocity = self._key_pan_initial_velocity, deaccelaration=self._key_pan_deacceleration)
        elif symbol == WispKey.UP:
            self.start_pan(pan_direction='up',
                           velocity=self._key_pan_initial_velocity, deaccelaration=self._key_pan_deacceleration)
        elif symbol == WispKey.DOWN:
            self.start_pan(pan_direction='down',
                           velocity=self._key_pan_initial_velocity, deaccelaration=self._key_pan_deacceleration)

    def start_pan(self, pan_direction, velocity=None, deaccelaration=None):
        self.start_interaction(f'pan_{pan_direction}')
        self._current_pan_velocity = velocity
        self._current_pan_deacceleration = deaccelaration
        self._current_pan_direction = pan_direction
        self._remaining_pan_time = abs(self._current_pan_velocity / self._current_pan_deacceleration)
        if pan_direction in ('left', 'down', 'backward'):  # Directions that move opposite to camera axes
            self._current_pan_velocity *= -1

    def zoom(self, amount):
        if self.camera.lens_type == 'ortho':
            # Under orthographic projection, objects are not affected by distance to the camera
            amount = self._zoom_ortho_distance_weight * self.camera.fov_distance * abs(amount) * np.sign(amount)
            self.camera.zoom(amount)
            # Keep distance at reasonable range
            self.camera.fov_distance = torch.clamp(self.camera.fov_distance, *self._zoom_ortho_fov_dist_range)
        else:
            dist = self.camera.cam_pos().norm()
            amount *= self._zoom_persp_distance_weight * dist
            self.camera.move_forward(amount)

    def progress_pan(self, dt):
        if self._current_pan_direction is None or self._current_pan_velocity == 0:
            return
        dt = min(self._remaining_pan_time, dt)
        pos_delta = dt * self._current_pan_velocity
        if self._current_pan_direction in ('forward', 'backward'):
            cam_pos, cam_forward = self.camera.cam_pos().squeeze(), self.camera.cam_forward().squeeze()
            new_pos = (cam_pos + cam_forward * pos_delta)
            if ('xz' in self.planes_forbidden_zooming_through and new_pos[1].sign() * cam_pos[1].sign() == -1) or \
                ('xy' in self.planes_forbidden_zooming_through and new_pos[2].sign() * cam_pos[2].sign() == -1) or \
                ('yz' in self.planes_forbidden_zooming_through and new_pos[0].sign() * cam_pos[0].sign() == -1):
                self._remaining_pan_time = 0
            else:
                self.zoom(pos_delta)
        elif self._current_pan_direction in ('right', 'left'):
            dist = self.camera.cam_pos().norm()
            pos_delta *= self._key_pan_distance_weight * dist
            self.camera.move_right(pos_delta)
        elif self._current_pan_direction in ('up', 'down'):
            dist = self.camera.cam_pos().norm()
            pos_delta *= self._key_pan_distance_weight * dist
            self.camera.move_up(pos_delta)
        velocity_sign = np.sign(self._current_pan_velocity)
        deaccel_amount = velocity_sign * self._current_pan_deacceleration * dt
        self._current_pan_velocity -= deaccel_amount
        self._remaining_pan_time = max(0, self._remaining_pan_time - dt)
        if np.sign(self._current_pan_velocity) != velocity_sign or \
                self._current_pan_velocity == 0 or self._remaining_pan_time == 0:
            self.end_pan()

    def end_pan(self):
        self.end_interaction()
        if not self.is_interacting():   # End only if other interactions have not taken place meanwhile
            self._current_pan_velocity = 0
            self._current_pan_direction = None
            self._remaining_pan_time = 0

    def handle_key_release(self, symbol, modifiers):
        pass

    def handle_mouse_press(self, x, y, button):
        self.start_interaction('pan_withmouse')

    def handle_mouse_drag(self, x, y, dx, dy, button):
        dist_normalization = self._mouse_pan_distance_weight * self.camera.cam_pos().norm()
        self.camera.move_right(dist_normalization * -dx)
        self.camera.move_up(dist_normalization * dy)

    def handle_mouse_release(self, x, y, button):
        self.end_pan()

    def handle_mouse_motion(self, x, y, dx, dy):
        """ The mouse was moved with no buttons held down. """
        pass

    def start_interaction(self, interaction_id):
        self.interactions_stack.append(interaction_id)

    def end_interaction(self):
        # On some occassions the app may be out of focus and some interactions are not registered properly.
        # Silently ignore that.
        if len(self.interactions_stack) > 0:
            self.interactions_stack.pop()

    def stop_all_current_interactions(self):
        while self.is_interacting():
            last_interaction_started = self.get_last_interaction_started()
            if last_interaction_started.startswith('pan'):
                self.end_pan()
            else:
                self.end_interaction()

    def is_interacting(self):
        return len(self.interactions_stack) > 0

    def get_last_interaction_started(self):
        return self.interactions_stack[-1] if self.is_interacting() else None

    def has_interaction(self, interaction_id):
        return interaction_id in self.interactions_stack

    @property
    def camera(self):
        return self.render_core.camera
