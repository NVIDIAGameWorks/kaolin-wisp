# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch


def normalize_pointcloud(coords, return_scale=False):
    """Normalizes pointcloud to an AABB within [-1, 1].

    Args:
        coords (torch.FloatTensor): 3D coordinates of shape [N, 3]
        return_scale (bool): If True, will return the center of the cloud and the scaling factor.

    Returns:
        (torch.FloatTensor) or (torch.FloatTensor, torch.FloatTensor, float):
        - Normalized 3D coordinates of shape [N, 3]
        - Center of the cloud of shape [3]
        - Scaling factor (floating point value)
    """
    coords_max, _ = torch.max(coords, dim=0)
    coords_min, _ = torch.min(coords, dim=0)
    coords_center = (coords_max + coords_min) / 2.0

    # AABB normalize
    coords = coords - coords_center
    max_dist = torch.max(coords)
    coords_scale = 1.0 / max_dist
    coords *= coords_scale

    if return_scale:
        return coords, coords_center, coords_scale
    else:
        return coords
