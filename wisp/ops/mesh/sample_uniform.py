# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch

def sample_uniform(num_samples : int):
    """Sample uniformly in [-1,1] bounding volume.
    
    Args:
        num_samples(int) : number of points to sample
    
    Returns:
        (torch.FloatTensor): samples of shape [num_samples, 3]
    """
    return torch.rand(num_samples, 3) * 2.0 - 1.0

