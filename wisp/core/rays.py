# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from dataclasses import dataclass
import torch

# Largest representable float in this system
INFINITY = torch.finfo().max


@dataclass
class Rays:
    origins: torch.Tensor        # Ray's origin
    dirs: torch.Tensor           # Ray's normalized direction
    dist_min: float = 0.0                 # Distance in which ray intersection test begins
    dist_max: float = INFINITY            # Distance in which ray intersection test ends

    def __len__(self):
        return self.origins.shape[0]

    @property
    def shape(self):
        return self.origins.shape[:-1]

    @property
    def ndim(self):
        return self.origins.ndim - 1

    @classmethod
    def cat(cls, rays_list, dim=0):
        if dim < 0:
            dim -= 1
        if dim > rays_list[0].ndim - 1 or dim < -rays_list[0].ndim:
            raise IndexError("Dimension out of range (expected to be in range of "
                             f"[{-rays_list[0].ndim}, {rays_list[0].ndim-1}, but got {dim})")
        return Rays(
            origins=torch.cat([rays.origins for rays in rays_list], dim=dim),
            dirs=torch.cat([rays.dirs for rays in rays_list], dim=dim),
            dist_min=min([rays.dist_min for rays in rays_list]),
            dist_max=max([rays.dist_max for rays in rays_list])
        )

    @classmethod 
    def stack(cls, rays_list, dim=0):
        return Rays(
            origins=torch.stack([rays.origins for rays in rays_list], dim=dim),
            dirs=torch.stack([rays.dirs for rays in rays_list], dim=dim),
            dist_min=min([rays.dist_min for rays in rays_list]),
            dist_max=max([rays.dist_max for rays in rays_list])
        )

    def __getitem__(self, idx):
        return Rays(origins=self.origins[idx],
                   dirs=self.dirs[idx],
                   dist_min=self.dist_min,
                   dist_max=self.dist_max)
    
    def __len__(self):
        if self.origins.shape != self.dirs.shape:
            raise Exception(f"Rays.origins shape should match Rays.dirs shape, but got {self.origins.shape} and {self.dirs.shape}.")
        return self.origins.shape[0]

    def split(self, split_size):
        zipped = zip(torch.split(self.origins, split_size), torch.split(self.dirs, split_size))
        return [Rays(origins=origins, dirs=dirs, dist_min=self.dist_min, dist_max=self.dist_max) for origins, dirs in zipped]

    def reshape(self, *dims):
        return Rays(origins=self.origins.reshape(*dims),
                   dirs=self.dirs.reshape(*dims),
                   dist_min=self.dist_min,
                   dist_max=self.dist_max)

    def squeeze(self, dim):
        return Rays(origins=self.origins.squeeze(dim),
                   dirs=self.dirs.squeeze(dim),
                   dist_min=self.dist_min,
                   dist_max=self.dist_max)

    def contiguous(self):
        return Rays(origins=self.origins.contiguous(),
                   dirs=self.dirs.contiguous(),
                   dist_min=self.dist_min,
                   dist_max=self.dist_max)

    def to(self, *args, **kwargs) -> Rays:
        """ Cast to a different device / dtype """

        origins = self.origins.to(*args, **kwargs)
        dirs = self.dirs.to(*args, **kwargs)
        if origins is not self.origins or dirs is not self.dirs:
            return Rays(
                origins=origins,
                dirs=dirs,
                dist_min=self.dist_min,
                dist_max=self.dist_max
            )
        else:
            return self
