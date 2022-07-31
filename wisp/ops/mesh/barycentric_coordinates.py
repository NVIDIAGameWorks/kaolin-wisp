# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import numpy as np

# Same API as https://github.com/libigl/libigl/blob/main/include/igl/barycentric_coordinates.cpp

def barycentric_coordinates(
    points : torch.Tensor, 
    A : torch.Tensor,
    B : torch.Tensor,
    C : torch.Tensor):
    """
    Return barycentric coordinates for a given set of points and triangle vertices

    Args:
        points (torch.FloatTensor): [N, 3]
        A (torch.FloatTensor): [N, 3] vertex0
        B (torch.FloatTensor): [N, 3] vertex1
        C (torch.FloatTensor): [N, 3] vertex2
    
    Returns:
        (torch.FloatTensor): barycentric coordinates of [N, 2] 
    """

    v0 = B-A
    v1 = C-A
    v2 = points-A
    d00 = (v0*v0).sum(dim=-1)
    d01 = (v0*v1).sum(dim=-1)
    d11 = (v1*v1).sum(dim=-1)
    d20 = (v2*v0).sum(dim=-1)
    d21 = (v2*v1).sum(dim=-1)
    denom = d00*d11 - d01*d01
    L = torch.zeros(points.shape[0], 3, device=points.device)
    # Warning: This clipping may cause undesired behaviour
    L[...,1] = torch.clip((d11*d20 - d01*d21)/denom, 0.0, 1.0)
    L[...,2] = torch.clip((d00*d21 - d01*d20)/denom, 0.0, 1.0)
    L[...,0] = torch.clip(1.0 - (L[...,1] + L[...,2]), 0.0, 1.0)
    return L
