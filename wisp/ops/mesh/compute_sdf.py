# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import math
import contextlib
import os
import sys

import torch
import numpy as np
import wisp._C as _C

def compute_sdf(
    V : torch.Tensor,
    F : torch.Tensor,
    points : torch.Tensor,
    split_size : int = 10**6):
    """Computes SDF given point samples and a mesh.

    Args:
        V (torch.FloatTensor): #V, 3 array of vertices
        F (torch.LongTensor): #F, 3 array of indices
        points (torch.FloatTensor): [N, 3] array of points to sample
        split_size (int): The batch at which the SDF will be computed. The kernel will break for too large
                          batches; when in doubt use the default.

    Returns:
        (torch.FloatTensor): [N, 1] array of computed SDF values.
    """
    mesh = V[F]

    _points = torch.split(points, split_size)
    sdfs = []
    for _p in _points:
        sdfs.append(_C.external.mesh_to_sdf_cuda(_p.cuda().contiguous(), mesh.cuda().contiguous())[0])
    return torch.cat(sdfs)[...,None]
