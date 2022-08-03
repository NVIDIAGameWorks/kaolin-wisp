# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch


def total_variation(pidx, trinkets, features, level):
    """Calculates total variation for the voxels specified by the pidx.

    Args:
        pidx : int tensor of size [N] specifying the point indices to calculate TV on.
        trinkets : the trinkets.
        features : the features for the given level. (assumes the correct level is given)
        level : int specifying the level of spc

    Returns:
        (torch.FloatTensor): Total variation of shape [1]
    """
    resolution = 2**level
    # N, 8, F tensor of features
    feats = features[trinkets.index_select(0, pidx).long()]

    # N, F
    diff_x = ((torch.abs(feats[:, [4,5,6,7]] - feats[:, [0,1,2,3]]) / resolution)**2).sum((1,2))
    diff_y = ((torch.abs(feats[:, [2,3,6,7]] - feats[:, [0,1,4,5]]) / resolution)**2).sum((1,2))
    diff_z = ((torch.abs(feats[:, [1,3,5,7]] - feats[:, [0,2,4,6]]) / resolution)**2).sum((1,2)) 
    return diff_x + diff_y + diff_z
