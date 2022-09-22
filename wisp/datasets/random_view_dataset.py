# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from typing import Callable, Tuple, Union
from copy import deepcopy
import unittest
import random
import numpy as np

import torch

from kaolin.render.camera import Camera
from kaolin.render.camera.extrinsics import CameraExtrinsics
from torch.utils.data import Dataset
from wisp.utils import DotDict
from wisp.ops.raygen import generate_pinhole_rays, generate_centered_pixel_coords


def spherical_eye(
    radius=1,
    theta=np.pi / 3,
    phi=0,
):
    return torch.FloatTensor(
        [
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta),
            radius * np.sin(theta) * np.cos(phi),
        ],
    )  # [3]


# TODO (cchoy): move the radius/theta initialization to kaolin CameraExtrinsics
def spherical_coord_to_pose(
    radius=1, theta=np.pi / 3, phi=0, up=torch.FloatTensor([0, 1, 0])
):
    """generate camera pose from a spherical coordinate

    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4] in OpenGL convention
    """

    eye = spherical_eye(radius, theta, phi)

    # lookat
    def normalize(vec):
        return torch.nn.functional.normalize(vec, dim=-1)

    backward = -normalize(eye)
    right = normalize(torch.cross(backward, up, dim=-1))
    up = normalize(torch.cross(right, backward, dim=-1))

    world_rot = torch.stack((right, up, -backward), dim=1)
    world_tran = -world_rot @ eye.unsqueeze(-1)

    return CameraExtrinsics._from_world_in_cam_coords(
        rotation=world_rot, translation=world_tran, device="cpu", requires_grad=False
    )


class RandomViewDataset(Dataset):
    def __init__(
        self,
        # TODO(cchoy) add different random view types e.g. forward_facing, inward
        n_size=100,  # length of this dataset. Used to define number of iterations per epoch
        view_radius_range: Tuple = (2, 4),
        view_theta_range: Tuple = (np.pi / 4, np.pi / 2 - np.pi / 8),
        view_phi_range: Tuple = (0, 2 * np.pi),
        viewport_height: int = 320,
        viewport_width: int = 320,
        fov: float = 30 * np.pi / 180,
        ray_dist_range: Tuple = (0.01, 8),
        look_at: Tuple = (0, 0, 0),
        num_rays: int = -1,  # number of rays. If -1, return all rays
        transform: Callable = None,
        **kwargs,
    ):
        self.n_size = n_size
        self.cam = DotDict(
            dict(
                fov=fov,
                width=viewport_width,
                height=viewport_height,
            )
        )
        self.view_radius_range = view_radius_range
        self.view_theta_range = view_theta_range
        self.view_phi_range = view_phi_range
        self.ray_dist_range = ray_dist_range
        assert len(look_at) == 3
        self.look_at = look_at

        self.num_rays = num_rays
        self.transform = transform

    def __len__(self):
        """Length of the dataset in number of rays."""
        return self.n_size

    def __getitem__(self, idx: int):
        """Returns a ray."""
        # TODO (cchoy): uniform sphere sampling (http://corysimon.github.io/articles/uniformdistn-on-sphere/)
        radius = random.uniform(*self.view_radius_range)
        theta = random.uniform(*self.view_theta_range)
        phi = random.uniform(*self.view_phi_range)
        cam = Camera.from_args(
            eye=spherical_eye(radius, theta, phi),
            at=torch.tensor([0.0, 0.0, 0.0]),
            up=torch.tensor([0.0, 1.0, 0.0]),
            fov=self.cam.fov,
            width=self.cam.width,
            height=self.cam.height,
            device="cpu",
        )

        ray_grid = generate_centered_pixel_coords(
            cam.width, cam.height, cam.width, cam.height, device="cpu"
        )
        out = DotDict(dict(rays=generate_pinhole_rays(cam, ray_grid), cam=cam))
        if self.num_rays > 0:
            ray_idx = random.sample(range(len(out.rays)), self.num_rays)
            out.rays = out.rays[ray_idx]

        if self.transform is not None:
            out = self.transform(out)

        return out


class TestRandViewDataset(unittest.TestCase):
    def load(self):
        dataset = RandomViewDataset()
        print(dataset[0])
