# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import kaolin.ops.spc as spc_ops


def dilate_points(points, level):
    """Dilates the octree points.

    Args:
        points (torch.ShortTensor): The SPC points from some level
        level (int): The level from which the points come from

    Returns:
        (torch.ShortTensor): The dilated points
    """
    _x = torch.ShortTensor([[1 ,0 ,0]]).to(points.device)
    _y = torch.ShortTensor([[0 ,1 ,0]]).to(points.device)
    _z = torch.ShortTensor([[0 ,0 ,1]]).to(points.device)
    points = torch.cat([
        points + _x, points - _x,
        points + _y, points - _y,
        points + _z, points - _z,
        points + _x + _y, points + _x - _y, points + _x + _z, points + _x - _z,
        points + _y + _x, points + _y - _x, points + _y + _z, points + _y - _z,
        points + _z + _x, points + _z - _x, points + _z + _y, points + _z - _y,
        points + _x + _y + _z, points + _x + _y - _z,
        points + _x - _y + _z, points + _x - _y - _z,
        points - _x + _y + _z, points - _x + _y - _z,
        points - _x - _y + _z, points - _x - _y - _z,
        ], dim=0)
    points = torch.clip(points, 0, 2** level - 1)

    unique, unique_keys, unique_counts = torch.unique(points.contiguous(), dim=0,
                                                      return_inverse=True, return_counts=True)

    morton, keys = torch.sort(spc_ops.points_to_morton(unique.contiguous()).contiguous())

    points = spc_ops.morton_to_points(morton.contiguous())

    return points