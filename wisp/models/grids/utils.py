# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn
from typing import Tuple, Optional

class MultiTable(nn.Module):
    """Class that holds multiresolution grid tables.
    """

    def __init__(
        self, 
        resolutions : Tuple[int, ...], 
        coord_dim   : int, 
        feature_dim : int, 
        std         : float             = 0.01, 
        max_feats   : Optional[int]     = None, 
    ):
        """
        Args:
            resolutions (List[int, ...]): The resolutions in the multiresolution hierarchy.
            coord_dim (int): The coordinate dimension for the grid.
            feature_dim (int): The feature dimension for the grid.
            std (float): The standard deviation for the features.
            max_feats (Optional[int]): The max number of features (when in use for hash grids, for example)
        """
        super().__init__()

        self.num_lods = len(resolutions)
        self.max_feats = max_feats
        
        self.register_buffer("begin_idxes", torch.zeros(self.num_lods+1, dtype=torch.int64))
        self.register_buffer("num_feats", torch.zeros(self.num_lods, dtype=torch.int64))

        self.coord_dim = coord_dim
        self.feature_dim = feature_dim

        self.resolutions = torch.zeros([self.num_lods, 1], dtype=torch.int64)
        for i in range(len(resolutions)):
            self.resolutions[i] = resolutions[i]
        
        num_so_far = 0
        for i in range(self.num_lods):
            resolution = self.resolutions[i]
            num_feats_level = resolution[0] ** self.coord_dim
            
            if self.max_feats:
                num_feats_level = min(self.max_feats, num_feats_level)
            
            self.begin_idxes[i] = num_so_far
            self.num_feats[i] = num_feats_level
            num_so_far += num_feats_level

        self.begin_idxes[self.num_lods] = num_so_far

        self.total_feats = sum(self.num_feats)
        self.feats = nn.Parameter(torch.randn(self.total_feats, self.feature_dim) * std)

    def get_level(self, idx):
        """Gets the features for a specific level.

        Args:
            idx (int): The level of the multiresolution grid to get.
        """
        return self.feats[self.begin_idxes[idx]:self.begin_idxes[idx+1]]
