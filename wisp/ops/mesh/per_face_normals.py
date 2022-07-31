# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch

def per_face_normals(
    V : torch.Tensor,
    F : torch.Tensor):
    """Compute normals per face.
    
    Args:
        V (torch.FloatTensor): Vertices of shape [V, 3]
        F (torch.LongTensor): Faces of shape [F, 3]
    
    Returns:
        (torch.FloatTensor): Normals of shape [F, 3]
    """
    mesh = V[F]

    vec_a = mesh[:, 0] - mesh[:, 1]
    vec_b = mesh[:, 1] - mesh[:, 2]
    normals = torch.cross(vec_a, vec_b)
    return normals

