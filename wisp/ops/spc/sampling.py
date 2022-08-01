# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch


def sample_spc(
    corners     : torch.Tensor,
    level       : int,
    num_samples : int):
    """Sample uniformly in [-1,1] bounding volume within SPC voxels

    Args:
        corners (tensor)  : set of corners to sample from
        level (int)         : level to sample from
        num_samples (int) : number of points to sample

    Returns:
        (torch.FloatTensor): samples of shape [num_samples, 3]
    """

    res = 2.0**level
    samples = torch.rand(corners.shape[0], num_samples, 3, device=corners.device)
    samples = corners[...,None,:3] + samples
    samples = samples.reshape(-1, 3)
    samples /= res
    return samples * 2.0 - 1.0


def sample_from_depth_intervals(depth_intervals, num_samples):
    """Convert depth intervals to samples.

    SPC raytrace will return a [num_nuggets, 2] array where the first element is the entry depth
    and the second element is the exit depth. This function will convert them into a
    [num_nuggets, num_samples, 3] array of samples.

    Args:
        depth_intervals (torch.FloatTensor): intervals of shape [num_nuggets, 2]
        num_samples (int): sample size

    Returns:
        (torch.FloatTensor): Samples of shape [num_nuggets, num_samples, 3]
    """
    device = depth_intervals.device
    steps = torch.arange(num_samples, device=device)[None].float().repeat([depth_intervals.shape[0], 1])
    steps += torch.rand_like(steps)
    steps *= (1.0 / num_samples)
    samples = depth_intervals[..., 0:1] + (depth_intervals[..., 1:2] - depth_intervals[..., 0:1]) * steps

    return samples


def expand_pack_boundary(pack_boundary, num_samples):
    """Expands the pack boundaries according to the number of samples.

    Args:
        pack_boundary (torch.BoolTensor): pack boundaries [N]
        num_samples (int): Number of samples

    Returns:
        (torch.BoolTensor): pack boundaries of shape [N*num_samples]
    """
    bigpack_boundary = torch.zeros(pack_boundary.shape[0]*num_samples, device=pack_boundary.device).bool()
    bigpack_boundary[pack_boundary.nonzero().long() * num_samples] = True
    bigpack_boundary = bigpack_boundary.int()
    return bigpack_boundary
