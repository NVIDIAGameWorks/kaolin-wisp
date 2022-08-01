# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import numpy as np
import kaolin.ops.spc as spc_ops


def create_dense_octree(level):
    """Creates a dense SPC model

    Args:
        level (int): The level at which the octree will be initialized to.

    Returns:
        (torch.ByteTensor): the octree tensor
    """
    coords = np.arange(2**level)
    points = np.array(np.meshgrid(coords, coords, coords, indexing='xy'))
    points = points.transpose(3,2,1,0).reshape(-1, 3)
    points = torch.from_numpy(points).short().cuda()
    octree = spc_ops.unbatched_points_to_octree(points, level)
    return octree


def make_trilinear_spc(points, pyramid):
    """Builds a trilinear spc from a regular spc.

    Args:
        points (torch.ShortTensor): The point_hierarchy.
        pyramid (torch.LongTensor): The pyramid tensor.

    Returns:
        (torch.ShortTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor)
        - The dual point_hierarchy.
        - The dual pyramid.
        - The trinkets.
        - The parent pointers.
    """
    points_dual, pyramid_dual = spc_ops.unbatched_make_dual(points, pyramid)
    trinkets, parents = spc_ops.unbatched_make_trinkets(points, pyramid, points_dual, pyramid_dual)
    return points_dual, pyramid_dual, trinkets, parents
