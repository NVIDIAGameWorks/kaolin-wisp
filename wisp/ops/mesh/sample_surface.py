# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from .random_face import random_face
from .area_weighted_distribution import area_weighted_distribution

def sample_surface(
    V : torch.Tensor,
    F : torch.Tensor,
    num_samples : int,
    distrib = None):
    """Sample points and their normals on mesh surface.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        F (torch.Tensor): #F, 3 array of indices
        num_samples (int): number of surface samples
        distrib: distribution to use. By default, area-weighted distribution is used
    
    Returns:
        (torch.FloatTensor): samples of shape [num_samples, 3]
    """
    if distrib is None:
        distrib = area_weighted_distribution(V, F)

    # Select faces & sample their surface
    fidx, normals = random_face(V, F, num_samples, distrib)
    f = V[fidx]

    u = torch.sqrt(torch.rand(num_samples)).to(V.device).unsqueeze(-1)
    v = torch.rand(num_samples).to(V.device).unsqueeze(-1)

    samples = (1 - u) * f[:,0,:] + (u * (1 - v)) * f[:,1,:] + u * v * f[:,2,:]
    
    return samples, normals

