# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from wisp.core.colors import white, black, red, green, blue
from typing import Dict
from kaolin.render.camera import Camera
from wisp.core.primitives import PrimitivesPack
from wisp.gfx.datalayers import Datalayers


class CameraDatalayers(Datalayers):
    def needs_redraw(self) -> bool:
        raise True

    def regenerate_data_layers(self, cameras: Dict[str, Camera],
                               bg_color: torch.Tensor = None) -> Dict[str, PrimitivesPack]:
        if bg_color == white:
            color = torch.tensor((*black, 1.0))
        else:
            color = torch.tensor((*white, 1.0))

        red_rgba = torch.tensor((*red, 1.0))
        green_rgba = torch.tensor((*green, 1.0))
        blue_rgba = torch.tensor((*blue, 1.0))

        layers = dict()

        for camera_id, camera in cameras.items():
            camera_outlines = PrimitivesPack()
            origin = camera.cam_pos().squeeze()
            forward = camera.cam_forward().squeeze()
            right = camera.cam_right().squeeze()
            up = camera.cam_up().squeeze()
            forward_length = 0.2
            right_length = 0.2
            up_length = 0.2
            corner1 = origin - forward * forward_length + right * right_length + up * up_length
            corner2 = origin - forward * forward_length - right * right_length + up * up_length
            corner3 = origin - forward * forward_length + right * right_length - up * up_length
            corner4 = origin - forward * forward_length - right * right_length - up * up_length
            camera_outlines.add_lines(start=origin, end=corner1, color=color)
            camera_outlines.add_lines(start=origin, end=corner2, color=color)
            camera_outlines.add_lines(start=origin, end=corner3, color=color)
            camera_outlines.add_lines(start=origin, end=corner4, color=color)
            camera_outlines.add_lines(start=corner1, end=corner2, color=color)
            camera_outlines.add_lines(start=corner3, end=corner4, color=color)
            camera_outlines.add_lines(start=corner2, end=corner4, color=color)
            camera_outlines.add_lines(start=corner1, end=corner3, color=color)

            # Draw axes above camera
            above = origin - forward * forward_length + up * up_length * 1.1
            cam_axes = camera.R[0]  # Returns the camera axes as rows
            permuted_axes = cam_axes.T @ camera.extrinsics.basis_change_matrix
            cam_right = permuted_axes[:, 0]
            cam_up = permuted_axes[:, 1]
            cam_forward = permuted_axes[:, 2]
            camera_outlines.add_lines(start=above, end=above + cam_right * 0.1, color=red_rgba)
            camera_outlines.add_lines(start=above, end=above + cam_up * 0.1, color=green_rgba)
            camera_outlines.add_lines(start=above, end=above + cam_forward * 0.1, color=blue_rgba)
            layers[camera_id] = camera_outlines
        return layers

