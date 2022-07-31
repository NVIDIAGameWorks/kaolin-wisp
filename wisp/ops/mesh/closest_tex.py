# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import numpy as np
from .barycentric_coordinates import barycentric_coordinates
from .closest_point import closest_point
from .sample_tex import sample_tex

def closest_tex(
    V : torch.Tensor, 
    F : torch.Tensor,
    TV : torch.Tensor,
    TF : torch.Tensor,
    materials,
    points : torch.Tensor):
    """Returns the closest texture for a set of points.

        V (torch.FloatTensor): mesh vertices of shape [V, 3] 
        F (torch.LongTensor): mesh face indices of shape [F, 3]
        TV (torch.FloatTensor): 
        TF (torch.FloatTensor):
        materials:
        points (torch.FloatTensor): sample locations of shape [N, 3]

    Returns:
        (torch.FloatTensor): texture samples of shape [N, 3]
    """

    TV = TV.to(V.device)
    TF = TF.to(V.device)
    points = points.to(V.device)

    dist, hit_pts, hit_tidx = closest_point(V, F, points)
    
    hit_F = F[hit_tidx]
    hit_V = V[hit_F]
    BC = barycentric_coordinates(hit_pts.cuda(), hit_V[:,0], hit_V[:,1], hit_V[:,2])

    hit_TF = TF[hit_tidx]
    hit_TM = hit_TF[...,3]
    hit_TF = hit_TF[...,:3]

    if TV.shape[0] > 0:
        hit_TV = TV[hit_TF]
        hit_Tp = (hit_TV * BC.unsqueeze(-1)).sum(1)
    else:
        hit_Tp = BC
    
    rgb = sample_tex(hit_Tp, hit_TM, materials)

    return rgb, hit_pts, dist
