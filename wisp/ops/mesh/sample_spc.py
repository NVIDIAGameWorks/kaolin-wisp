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

