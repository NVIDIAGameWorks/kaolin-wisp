# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from .per_face_normals import per_face_normals

def area_weighted_distribution(
    V : torch.Tensor,
    F : torch.Tensor, 
    normals : torch.Tensor = None):
    """Construct discrete area weighted distribution over triangle mesh.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        F (torch.Tensor): #F, 3 array of indices
        normals (torch.Tensor): normals (if precomputed)
        eps (float): epsilon
    
    Returns:
        (torch.distributions): Distribution to be used
    """

    if normals is None:
        normals = per_face_normals(V, F)
    areas = torch.norm(normals, p=2, dim=1) * 0.5
    areas /= torch.sum(areas) + 1e-10
    
    # Discrete PDF over triangles
    return torch.distributions.Categorical(areas.view(-1))

